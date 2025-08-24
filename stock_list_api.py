from flask import Flask, request, jsonify, render_template_string, url_for
import pandas as pd
import datetime
import time
import json
import ssl
import re
from dateutil.relativedelta import relativedelta
import requests
import numpy as np

def clean_query(query_str):
    return query_str.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


app = Flask(__name__)
@app.route('/', methods=['POST'])
def echo():

    data = request.get_json()
    json_string = data.get('stock_json')
    # df = pd.read_excel('s&p500_data.xlsx')
    df = pd.read_excel('data2.xlsx')
    # df['代碼'] = [item.split()[0] for item in df['代碼']]



    data = json.loads(json_string)
    combined_mask = pd.Series([True] * len(df))

    # 遍歷所有可能是 label 的欄位（只處理 list 結構）
    for label_group in data.values():
        if isinstance(label_group, list):
            for label in label_group:
                query_type = label.get("query_type", "")
                query = label.get("query", "")
                label_name = label.get("label_zh", "未命名標籤")

                label_mask = pd.Series([True] * len(df))  # 預設為全 True

                if query_type == "keyword" and query == "欄位中包含任一關鍵字":
                    columns = label.get("columns", [])
                    keywords = label.get("keywords", [])

                    label_mask = df[columns].apply(
                        lambda col: col.astype(str).apply(lambda val: any(keyword in val for keyword in keywords))
                    ).any(axis=1)

                else:
                    try:
                        cleaned_query = clean_query(query)
                        label_mask = eval(cleaned_query, {"df": df, "pd": pd})
                    except Exception as e:
                        print(f"⚠️ 無法執行「{label_name}」的 query：{e}")
                        continue

                # 將目前條件與累積條件取交集
                combined_mask &= label_mask
                print(f"✅ 套用「{label_name}」條件後剩下的資料列：")
                print(df[label_mask])

    final_result = df[combined_mask]
    pool = final_result['代碼'].tolist()

    return {"stock_list" : pool}

if __name__ == '__main__':
    app.run()