import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- 基礎設定與路徑 ---
st.set_page_config(page_title="Retail Strategy Hub", page_icon="📊", layout="wide")

# 自動定位數據路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'OnlineRetail.csv')

# --- 數據載入函數 ---
@st.cache_data
def load_data():
    # 讀取數據
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    
    # 1. 基礎清洗
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')] # 排除退貨
    df = df[df['Quantity'] > 0]
    
    # 2. 轉換日期與計算營收
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    # 3. 排除商業雜訊 (郵資、手續費、銀行費用等)
    noise_items = ['POSTAGE', 'DOTCOM POSTAGE', 'SERVICE', 'BANK CHARGES', 'AMAZON FEE']
    df = df[~df['Description'].str.upper().str.contains('|'.join(noise_items), na=False)]
    
    return df

# --- 啟動載入 ---
try:
    with st.spinner('🚀 正在處理 50 萬筆零售大數據，請稍候...'):
        df_all = load_data()
except Exception as e:
    st.error(f"❌ 數據載入失敗！請確認 OnlineRetail.csv 檔案路徑。錯誤：{e}")
    st.stop()

# --- 側邊欄 ---
st.sidebar.title("🛠 策略控制中心")
country_list = sorted(df_all['Country'].unique())
selected_country = st.sidebar.selectbox(
    "選擇分析市場", 
    country_list, 
    index=list(country_list).index('United Kingdom') if 'United Kingdom' in country_list else 0
)

# 過濾市場數據
df_filtered = df_all[df_all['Country'] == selected_country]

# --- 主要 UI 介面 ---
st.title("🛒 Retail Operation Center")
st.markdown(f"**當前市場： `{selected_country}`** | 數據規模： `{len(df_filtered):,}` 筆交易")

tab1, tab2, tab3 = st.tabs(["👥 RFM 客戶分析", "🛍️ 購物籃分析", "📈 市場概況"])

# --- Tab 1: RFM 分析 ---
with tab1:
    st.subheader("客戶價值分佈 (Recency, Frequency, Monetary)")
    snapshot = df_filtered['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_filtered.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary'})
    
    fig_rfm = px.scatter(
        rfm, x="Recency", y="Monetary", size="Frequency", color="Monetary",
        hover_name=rfm.index, log_y=True,
        title="RFM 客戶分佈圖 (圓點大小代表消費次數)",
        labels={'Recency': '距上次消費天數', 'Monetary': '累計消費金額 (log scale)'},
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

# --- Tab 2: 購物籃分析 (關鍵優化區) ---
with tab2:
    st.subheader("關聯銷售挖掘 (Association Rules)")
    
    # 限制運算量以確保效能
    if len(df_filtered) > 15000:
        st.warning(f"⚠️ 為確保運算效能，系統自動選取 {selected_country} 最新 15,000 筆紀錄。")
        df_basket_input = df_filtered.sort_values('InvoiceDate', ascending=False).head(15000)
    else:
        df_basket_input = df_filtered

    # 建立矩陣
    basket = (df_basket_input.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    
    # 語法修正：新版 Pandas 使用 .map 取代 .applymap
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
    
    with st.spinner('🧬 正在挖掘關聯規則...'):
        frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        col1, col2 = st.columns([1, 2])
        with col1:
            sel_a = st.selectbox("1️⃣ 選擇領頭商品 (Driver Item)：", sorted(rules['A'].unique()))
            top_rules = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).head(5)
            
            st.metric("最強關聯商品", top_rules['B'].iloc[0])
            st.metric("搭配購買提升率 (Lift)", f"{top_rules['lift'].iloc[0]:.2f}x")

        with col2:
            fig_rules = px.bar(
                top_rules, x="lift", y="B", orientation='h',
                title=f"買了 '{sel_a}' 後，最常順手買的 Top 5 商品",
                labels={'lift': '推薦權重 (Lift)', 'B': '建議加購商品'},
                color='lift', color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_rules, use_container_width=True)
            
        st.info("💡 **商業建議：** Lift 越高代表關聯性越強，建議將這些商品擺放在相鄰貨架或作為組合包銷售。")
    else:
        st.write("🔍 該市場目前尚無顯著的關聯規則，請嘗試更換市場或降低分析門檻。")

# --- Tab 3: 市場概況 ---
with tab3:
    st.subheader("營運關鍵指標 (KPIs)")
    c1, c2, c3 = st.columns(3)
    c1.metric("總訂單數", f"{df_filtered['InvoiceNo'].nunique():,}")
    c2.metric("總營收", f"£{df_filtered['TotalSum'].sum():,.0f}")
    c3.metric("平均客單價", f"£{df_filtered['TotalSum'].sum() / df_filtered['InvoiceNo'].nunique():,.2f}")
    
    # 營收趨勢
    df_filtered['Month'] = df_filtered['InvoiceDate'].dt.to_period('M').astype(str)
    trend = df_filtered.groupby('Month')['TotalSum'].sum().reset_index()
    fig_trend = px.line(trend, x='Month', y='TotalSum', title="月營收成長趨勢圖", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard engineered by Yi-Han.")
