import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- 基礎路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'OnlineRetail.csv')

st.set_page_config(page_title="Retail Big Data Hub", layout="wide")
st.title("🛒 Retail Operation Center (Big Data Edition)")

@st.cache_data
def load_data():
    # 讀取大檔案時顯示載入中
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    # 基礎清洗
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    return df

try:
    with st.spinner('🚀 正在載入 50 萬筆交易數據，請稍候...'):
        df_all = load_data()
except Exception as e:
    st.error(f"讀取失敗！請確認 OnlineRetail.csv 是否在專案目錄下。錯誤：{e}")
    st.stop()

# --- 側邊欄控制 ---
st.sidebar.header("市場與數據量控制")
country_list = df_all['Country'].unique()
selected_country = st.sidebar.selectbox("選擇分析市場", country_list, index=list(country_list).index('United Kingdom'))

# 核心：過濾市場數據
df_filtered = df_all[df_all['Country'] == selected_country]

# 功能分頁
tab1, tab2, tab3 = st.tabs(["👥 RFM 客戶分析", "🛍️ 購物籃分析", "📈 市場概況"])

with tab1:
    st.subheader(f"{selected_country} - 客戶價值分佈")
    snapshot = df_filtered['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_filtered.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSum': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary'})
    
    fig = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Recency",
                     title="RFM 分析 (橫軸: 距上次消費天數, 縱軸: 累計消費額)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("關聯銷售挖掘 (Apriori Algorithm)")
    
    # 大數據保護機制：如果該國交易超過 10000 筆，僅分析最近的 10000 筆，避免當機
    if len(df_filtered) > 10000:
        st.warning(f"⚠️ 由於 {selected_country} 數據量過大，系統自動選取最新 10,000 筆交易進行運算。")
        df_basket_input = df_filtered.sort_values('InvoiceDate', ascending=False).head(10000)
    else:
        df_basket_input = df_filtered

    # 矩陣轉換
    basket = (df_basket_input.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
    
    with st.spinner('演算法執行中...'):
        frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        sel_a = st.selectbox("選擇領頭商品 (Driver)：", sorted(rules['A'].unique()))
        res = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        
        c1, c2 = st.columns(2)
        c1.metric("最強關聯商品", res['B'])
        c2.metric("搭配購買提升率 (Lift)", f"{res['lift']:.2f}x")
    else:
        st.write("目前 Support 門檻下無顯著規則。")

with tab3:
    st.subheader("市場統計數據")
    st.write(f"總訂單數: {len(df_filtered['InvoiceNo'].unique())}")
    st.write(f"總營收: €{df_filtered['TotalSum'].sum():,.0f}")
