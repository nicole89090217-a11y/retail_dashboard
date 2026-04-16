import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- 基礎設定 ---
st.set_page_config(page_title="Retail Strategy Hub V2", page_icon="📊", layout="wide")

# --- 自動分類邏輯函數 ---
def assign_category(description):
    desc = str(description).upper()
    if 'BAG' in desc or 'LUNCH BOX' in desc or 'TOTE' in desc:
        return '包袋與午餐盒 (Bags & Lunch Boxes)'
    if 'BOTTLE' in desc or 'CUP' in desc or 'MUG' in desc or 'TEAPOT' in desc:
        return '飲具系列 (Drinkware)'
    if 'CHRISTMAS' in desc or 'XMAS' in desc or 'STOCKED' in desc:
        return '聖誕季節裝飾 (Christmas)'
    if 'LIGHT' in desc or 'CANDLE' in desc or 'LANTERN' in desc:
        return '燈具與香氛 (Lighting & Candles)'
    if 'KITCHEN' in desc or 'CUTLERY' in desc or 'CAKE' in desc or 'BOWL' in desc:
        return '餐廚用品 (Kitchen & Dining)'
    if 'VINTAGE' in desc or 'RETRO' in desc or 'DECORATION' in desc or 'SIGN' in desc:
        return '居家裝飾 (Home Decor)'
    if 'STATIONERY' in desc or 'CARD' in desc or 'PENCIL' in desc or 'NOTEBOOK' in desc:
        return '文具與卡片 (Stationery)'
    return '其他生活禮品 (General Gifts)'

# --- 數據載入與清洗 ---
@st.cache_data
def load_and_process_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'OnlineRetail.csv')
    
    # 讀取數據
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    
    # 1. 基礎清洗
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    df = df[df['Quantity'] > 0]
    
    # 2. 排除商業雜訊 (郵資等)
    noise_items = ['POSTAGE', 'DOTCOM POSTAGE', 'SERVICE', 'BANK CHARGES', 'AMAZON FEE']
    df = df[~df['Description'].str.upper().str.contains('|'.join(noise_items), na=False)]
    
    # 3. 執行品類歸併
    df['Category'] = df['Description'].apply(assign_category)
    
    # 4. 計算日期與營收
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    return df

# --- 啟動載入 ---
try:
    with st.spinner('🚀 正在進行品類大數據歸併與分析...'):
        df_all = load_and_process_data()
except Exception as e:
    st.error(f"❌ 數據載入失敗！錯誤原因：{e}")
    st.stop()

# --- 主要 UI 介面 ---
st.title("🛒 Retail Strategy Center: Category Insight")
st.markdown(f"**數據總量：** `{len(df_all):,}` 筆交易 | **分析維度：** 品類與關聯性")

# 側邊欄過濾
st.sidebar.title("🛠 策略控制面板")
selected_country = st.sidebar.selectbox("選擇市場", sorted(df_all['Country'].unique()), index=list(sorted(df_all['Country'].unique())).index('United Kingdom'))
df_filtered = df_all[df_all['Country'] == selected_country]

tab1, tab2, tab3 = st.tabs(["📈 品類表現分析", "🛍️ 購物籃關聯挖掘", "👥 客戶價值 (RFM)"])

# --- Tab 1: 品類表現分析 (新增) ---
with tab1:
    st.subheader(f"📊 {selected_country} 市場品類結構")
    
    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        # 品類營收占比
        category_revenue = df_filtered.groupby('Category')['TotalSum'].sum().reset_index()
        fig_pie = px.pie(category_revenue, values='TotalSum', names='Category', 
                         title='各品類營收貢獻比', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_bar:
        # 品類銷量排名
        category_qty = df_filtered.groupby('Category')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=True)
        fig_bar = px.bar(category_qty, x='Quantity', y='Category', orientation='h',
                         title='各品類總銷售件數', color='Quantity',
                         color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 2: 購物籃分析 (優化版) ---
with tab2:
    st.subheader("🛍️ 關聯銷售挖掘 (Association Rules)")
    
    # 為了運算效率，取該市場最新數據
    df_basket_input = df_filtered.sort_values('InvoiceDate', ascending=False).head(15000)
    
    # 建立購物籃矩陣
    basket = (df_basket_input.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
    
    with st.spinner('🧬 正在挖掘商品間的「真愛」關聯...'):
        frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        sel_a = st.selectbox("1️⃣ 搜尋領頭商品 (Driver)：", sorted(rules['A'].unique()))
        top_rules = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).head(5)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info(f"**分析結果：**\n\n當客戶購買了 **{sel_a}**，他們極有可能也會對右側圖表中的商品感興趣。")
            st.metric("最高提升倍率 (Lift)", f"{top_rules['lift'].iloc[0]:.2f}x")
        
        with c2:
            fig_rules = px.bar(top_rules, x='lift', y='B', orientation='h',
                               title=f"建議搭配 '{sel_a}' 銷售的商品",
                               labels={'lift': '購買機率提升倍率', 'B': '建議加購品項'},
                               color='lift', color_continuous_scale='Viridis')
            st.plotly_chart(fig_rules, use_container_width=True)
    else:
        st.warning("🔍 該市場目前樣本數不足以產生顯著關聯，請嘗試切換至 United Kingdom。")

# --- Tab 3: RFM 客戶分析 ---
with tab3:
    st.subheader("👥 客戶貢獻度分佈")
    snapshot = df_filtered['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_filtered.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary'})
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Monetary",
                         hover_name=rfm.index, log_y=True,
                         title="RFM 客戶分佈圖", labels={'Recency': '最後一次購買至今(天)'},
                         color_continuous_scale='Purples')
    st.plotly_chart(fig_rfm, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard engineered by Yi-Han.\n\n此版本已整合「品類歸併」邏輯。")
