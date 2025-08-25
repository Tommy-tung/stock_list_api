from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import re

def clean_query(q: str) -> str:
    return q.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

def iter_labels(payload):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for maybe_list in payload.values():
            if isinstance(maybe_list, list):
                for item in maybe_list:
                    if isinstance(item, dict):
                        yield item

def vectorized_keyword_mask(df: pd.DataFrame, columns, keywords):
    if not columns or not keywords:
        return pd.Series(False, index=df.index)
    safe_parts = [re.escape(k) for k in keywords if isinstance(k, str) and k.strip()]
    if not safe_parts:
        return pd.Series(False, index=df.index)
    pattern = re.compile("|".join(safe_parts), flags=re.IGNORECASE)
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        col_mask = df[col].astype("string").str.contains(pattern, na=False)
        mask = mask | col_mask
    return mask

def coerce_numeric(df: pd.DataFrame, cols):
    for col in cols or []:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    return df

def symbol_series(df: pd.DataFrame):
    sym_col = '代碼' if '代碼' in df.columns else '指数股路透代码' if '指数股路透代码' in df.columns else None
    if sym_col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[sym_col].astype(str).str.split().str[0]

# ---------- new utilities for relaxation ----------

def percentile_mask(df, col, p, direction="top"):
    """
    direction: 'top' -> >= quantile(p), 'bottom' -> <= quantile(p)
    p is in [0,1]
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    qv = s.quantile(p)
    if np.isnan(qv):
        return pd.Series(False, index=df.index)
    if direction == "top":
        return (s >= qv).fillna(False)
    else:
        return (s <= qv).fillna(False)

def build_label_mask(df, label):
    """
    Compute a boolean mask for one label. Returns (mask, meta) where meta carries info for relaxation.
    Supported query_type:
      - keyword
      - percentile (uses numeric percentile field)
      - percentile_and (percentiles: {col: pct})
      - condition / condition_and / condition_or (uses query string)
    """
    qtype = (label.get("query_type") or "").strip()
    query = clean_query(label.get("query", "") or "")
    name = label.get("label_zh", label.get("label_en", "未命名標籤"))
    cols = label.get("columns", []) or []

    # Try numeric coercion for relevant columns up front (safe for text)
    df = coerce_numeric(df, cols)

    meta = {
        "name": name,
        "qtype": qtype,
        "relaxable": False,        # can we soften this label?
        "percentiles": None,       # for percentile types
        "direction_map": None      # bottom/top per column
    }

    if qtype == "keyword":
        kws = label.get("keywords", []) or []
        mask = vectorized_keyword_mask(df, cols, kws)
        return mask, meta

    if qtype == "percentile":
        # Decide direction from value (30 -> bottom; 70 -> top)
        col = cols[0] if cols else None
        if not col:
            return pd.Series(False, index=df.index), meta
        pct = float(label.get("percentile", 50)) / 100.0
        direction = "bottom" if pct <= 0.5 else "top"
        mask = percentile_mask(df, col, pct, direction=direction)
        meta.update({"relaxable": True, "percentiles": {col: pct}, "direction_map": {col: direction}})
        return mask, meta

    if qtype == "percentile_and":
        percentiles = label.get("percentiles", {}) or {}
        if not percentiles:
            return pd.Series(False, index=df.index), meta
        current = pd.Series(True, index=df.index)
        direction_map = {}
        for col, perc in percentiles.items():
            p = float(perc) / 100.0
            direction = "bottom" if p <= 0.5 else "top"
            current &= percentile_mask(df, col, p, direction=direction)
            direction_map[col] = direction
        meta.update({"relaxable": True, "percentiles": {k: float(v)/100.0 for k, v in percentiles.items()},
                    "direction_map": direction_map})
        return current, meta

    # condition / condition_and / condition_or (or unknown -> treat as condition)
    try:
        env = {"df": df, "pd": pd, "np": np}
        if not query:
            # If query empty, accept all (safe fallback)
            mask = pd.Series(True, index=df.index)
        else:
            out = eval(query, {}, env)
            mask = out if isinstance(out, pd.Series) else pd.Series(bool(out), index=df.index)
            mask = mask.reindex(df.index, fill_value=False)
        # Hard to "soften" arbitrary expressions => treat as non‑relaxable
        return mask, meta
    except Exception:
        # Fail closed for this label only
        return pd.Series(False, index=df.index), meta

def k_of_n_mask(mask_dict, must=set(), k=None):
    """
    Combine masks with optional must-have set and k-of-n for the rest.
    mask_dict: {label_name: pd.Series}
    must: set of label names that must be satisfied (all)
    k: minimum number of remaining labels to be satisfied (if None, require all)
    """
    idx = next(iter(mask_dict.values())).index if mask_dict else None
    if idx is None:
        return None

    # must-have intersection
    if must:
        inter = pd.Series(True, index=idx)
        for m in must:
            msk = mask_dict.get(m)
            if msk is None:
                # missing must -> impossible
                return pd.Series(False, index=idx)
            inter &= msk
    else:
        inter = pd.Series(True, index=idx)

    # optional labels
    optional = [name for name in mask_dict.keys() if name not in must]
    if not optional:
        return inter

    stack = np.vstack([mask_dict[name].to_numpy(dtype=bool) for name in optional])  # shape: (n_opt, n_rows)
    counts = stack.sum(axis=0)  # satisfied count per row
    need = len(optional) if k is None else max(0, int(k))
    ok = counts >= need
    return inter & pd.Series(ok, index=idx)

def soften_percentiles(label_meta_list, step=0.05, floor=0.5, ceil=0.5):
    """
    Adjust percentiles:
      - For bottom-type: p := min(p + step, 1.0) but stop when p > floor_target (e.g., move 0.30 -> 0.35 -> 0.40 ...)
      - For top-type:    p := max(p - step, 0.0) but stop when p <  ceil_target (e.g., 0.70 -> 0.65 -> 0.60 ...)
    floor and ceil are 'stop lines' measured relative to 0.5. Default stops at 0.5 (neutral).
    Returns True if any change happened.
    """
    changed = False
    for meta in label_meta_list:
        if not meta.get("relaxable"):
            continue
        pcts = meta.get("percentiles") or {}
        dir_map = meta.get("direction_map") or {}
        for col, p in list(pcts.items()):
            direction = dir_map.get(col, "top")
            if direction == "bottom":
                # move upward toward 0.5/neutral (less strict)
                new_p = min(p + step, 1.0)
                # stop if we already crossed 0.5 (neutral)
                if p < 0.5 and new_p <= 0.5 + floor:
                    pcts[col] = new_p
                    changed = True
            else:  # top
                new_p = max(p - step, 0.0)
                if p > 0.5 and new_p >= 0.5 - ceil:
                    pcts[col] = new_p
                    changed = True
        meta["percentiles"] = pcts
    return changed

def recompute_masks_with_softened_percentiles(df, labels, metas):
    """
    Recompute masks for labels whose percentiles were softened.
    Returns updated mask_dict.
    """
    mask_dict = {}
    for label, meta in zip(labels, metas):
        name = meta["name"]
        qtype = meta["qtype"]
        if qtype == "percentile" and meta.get("percentiles"):
            col = label["columns"][0]
            p = list(meta["percentiles"].values())[0]
            direction = list(meta["direction_map"].values())[0]
            mask = percentile_mask(df, col, p, direction)
            mask_dict[name] = mask
        elif qtype == "percentile_and" and meta.get("percentiles"):
            current = pd.Series(True, index=df.index)
            for col, p in meta["percentiles"].items():
                direction = meta["direction_map"].get(col, "top")
                current &= percentile_mask(df, col, p, direction)
            mask_dict[name] = current
        else:
            # rebuild using original path so non‑relaxable labels remain intact
            mask, _ = build_label_mask(df, label)
            mask_dict[name] = mask
    return mask_dict

# ---------- API route with relaxation ----------

app = Flask(__name__)
@app.route('/', methods=['POST'])

def select_list():
    data = request.get_json(force=True) or {}
    json_string = data.get('stock_json', '[]')

    # optional runtime knobs
    min_results = int(data.get("min_results", 8))  # target pool size before relaxing further
    must_labels = set(data.get("must_labels", []))  # exact names (label_zh or label_en you pass in)
    k_floor     = int(data.get("k_floor", 1))       # lowest k in k-of-n
    soften_step = float(data.get("soften_step", 0.005))  # 0.5% step
    max_soften_rounds = int(data.get("max_soften_rounds", 12))  # e.g., 70 -> 50 in 12 steps

    # Load dataset
    df = pd.read_excel('data2.xlsx').copy()
    df['__SYMBOL__'] = symbol_series(df)
    
   # Parse label list
    try:
        labels_payload = json.loads(json_string)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON in 'stock_json': {e}", "stock_list": []}), 400

    labels = list(iter_labels(labels_payload))
    if not labels:
        # return jsonify({"stock_list": df['__SYMBOL__'].tolist(), "note": "No labels provided; returned all symbols."})
        return jsonify({"stock_list": []})

    # Build masks + metas (don’t combine yet)
    mask_dict = {}
    metas = []
    for label in labels:
        mask, meta = build_label_mask(df, label)
        name = meta["name"]
        mask_dict[name] = mask
        metas.append(meta)

    # --- Stage 0: strict AND of all labels
    strict_and = pd.Series(True, index=df.index)
    for m in mask_dict.values():
        strict_and &= m
    pool = df.loc[strict_and, '__SYMBOL__'].tolist()
    if len(pool) >= min_results:
        # return jsonify({"stock_list": pool, "relaxation": "strict_and"})
        return jsonify({"stock_list": pool})

    # --- Stage A: k-of-n (with must)
    # start from k = n_optional down to k_floor
    optional_names = [n for n in mask_dict.keys() if n not in must_labels]
    nopt = len(optional_names)
    for k in range(nopt, max(k_floor, 1) - 1, -1):
        mask_k = k_of_n_mask(mask_dict, must=must_labels, k=k)
        pool = df.loc[mask_k, '__SYMBOL__'].tolist()
        if len(pool) >= min_results:
            # return jsonify({"stock_list": pool, "relaxation": f"k_of_n (k={k}, must={list(must_labels)})"})
            return jsonify({"stock_list": pool})

    # --- Stage B: soften percentiles and retry k-of-n
    # progressively reduce strictness of percentile thresholds and recompute masks
    for r in range(max_soften_rounds):
        changed = soften_percentiles(metas, step=soften_step)
        if not changed:
            break
        mask_dict = recompute_masks_with_softened_percentiles(df, labels, metas)
        # try k from nopt down to k_floor again
        for k in range(nopt, max(k_floor, 1) - 1, -1):
            mask_k = k_of_n_mask(mask_dict, must=must_labels, k=k)
            pool = df.loc[mask_k, '__SYMBOL__'].tolist()
            if len(pool) >= min_results:
                # return jsonify({"stock_list": pool, "relaxation": f"soften_percentiles_round={r+1}, k_of_n (k={k}, must={list(must_labels)})"})
                return jsonify({"stock_list": pool})

    # --- Stage C: last resort OR of all (respecting must if provided)
    if must_labels:
        # must AND (OR of optional)
        base = pd.Series(True, index=df.index)
        for m in must_labels:
            base &= mask_dict.get(m, pd.Series(False, index=df.index))
        any_opt = pd.Series(False, index=df.index)
        for n in optional_names:
            any_opt |= mask_dict[n]
        final = base & any_opt
    else:
        final = pd.Series(False, index=df.index)
        for m in mask_dict.values():
            final |= m

    pool = df.loc[final, '__SYMBOL__'].tolist()
    # return jsonify({"stock_list": pool, "relaxation": "final_or"})  # Also reply the relaxation information
    return jsonify({"stock_list": pool})

if __name__ == '__main__':
    app.run()