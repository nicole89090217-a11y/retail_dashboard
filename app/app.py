import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- 基礎設定 ---
st.set_page_config(page_title="Retail Insights Center", page_icon="📈", layout="wide")

# --- 核心優化：深度分類邏輯 ---
def assigned_clean_category(description):
    desc = str(description).upper()
    # 建立更寬廣的歸併邏輯，減少類別數量
    if any(word in desc for word in ['BAG', 'TOTE', 'LUNCH BOX', 'CASE', 'BACKPACK']):
        return '包袋收納 (Bags & Cases)'
    if any(word in desc for word in ['CUP', 'MUG', 'BOTTLE', 'GLASS', 'TEAPOT', 'FLASK']):
        return '飲具系列 (Drinkware)'
    if any(word in desc for word in ['CHRISTMAS', 'XMAS', 'STOCKED', 'SNOW']):
        return '季節節慶 (Seasonal)'
    if any(word in desc for word in ['CANDLE', 'LIGHT', 'LANTERN', 'HOLDER', 'FAIRY']):
        return '燈具香氛 (Home Fragrance)'
    if any(word in desc for word in ['KITCHEN', 'DINING', 'BOWL', 'PLATE', 'CUTLERY', 'CAKE']):
        return '餐廚用品 (Kitchen)'
    if any(word in desc for word in ['DECORATION', 'ORNAMENT', 'VINTAGE', 'SIGN', 'FRAME', 'WALL']):
        return '居家風格 (Home Decor)'
    return '其他禮品 (General Gifts)'

# --- 數據載入與清洗 ---
@st.cache_data
def load_and_clean_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'OnlineRetail.csv')
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    
    # 基本過濾
    df.dropna(subset=['Description', 'CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    df = df[df['Quantity'] > 0]
    
    # 排除雜訊
    noise = ['POSTAGE', 'DOTCOM POSTAGE', 'SERVICE', 'BANK CHARGES', 'AMAZON FEE', 'SAMPLES']
    df = df[~df['Description'].str.upper().str.contains('|'.join(noise), na=False)]
    
    # 縮短商品名稱：只取前 25 個字，並刪除末尾空格，避免圖表過亂
    df['Short_Name'] = df['Description'].str[:25].str.strip()
    
    # 套用深度分類
    df['Category'] = df['Description'].apply(assigned_clean_category)
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    return df

try:
    with st.spinner('✨ 正在精煉零售數據...'):
        df_all = load_and_clean_data()
except Exception as e:
    st.error(f"❌ 數據載入失敗: {e}")
    st.stop()

# --- 側邊欄 ---
st.sidebar.title("💎 精煉控制台")
selected_country = st.sidebar.selectbox("選擇市場", sorted(df_all['Country'].unique()), index=list(sorted(df_all['Country'].unique())).index('United Kingdom'))
df_filtered = df_all[df_all['Country'] == selected_country]

# --- 主要 UI ---
st.title("📊 Retail Strategy Dashboard")
st.markdown(f"**市場分析：** `{selected_country}` | **數據日期範圍：** `{df_filtered['InvoiceDate'].min().date()}` 至 `{df_filtered['InvoiceDate'].max().date()}`")

tab1, tab2, tab3 = st.tabs(["💎 品類表現", "🔗 關聯分析", "📈 營運指標"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        # 品類營收
        cat_revenue = df_filtered.groupby('Category')['TotalSum'].sum().reset_index()
        fig_pie = px.pie(cat_revenue, values='TotalSum', names='Category', 
                         hole=0.5, title="各類別營收貢獻",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 熱銷品 Top 10 (使用縮短後的名稱)
        top_items = df_filtered.groupby('Short_Name')['Quantity'].sum().nlargest(10).reset_index().sort_values('Quantity')
        fig_bar = px.bar(top_items, x='Quantity', y='Short_Name', orientation='h',
                         title="本市場熱銷 Top 10 商品",
                         labels={'Short_Name': '商品名稱', 'Quantity': '銷量'},
                         color='Quantity', color_continuous_scale='Blues')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("🛍️ 購物籃深度關聯")
    # 增加運算筆數至 20,000 提高準確度
    df_basket_input = df_filtered.sort_values('InvoiceDate', ascending=False).head(20000)
    
    # 建立矩陣
    basket = (df_basket_input.groupby(['InvoiceNo', 'Short_Name'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
    
    with st.spinner('正在計算關聯規則...'):
        frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        sel_a = st.selectbox("請選擇一個商品，查看其最強搭配：", sorted(rules['A'].unique()))
        top_rules = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).head(5)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            best_match = top_rules.iloc[0]['B']
            st.success(f"**最佳搭配建議：**\n\n購買 **{sel_a}** 的客戶，極高機率也會購買 **{best_match}**。")
            st.metric("提升率 (Lift)", f"{top_rules.iloc[0]['lift']:.2f}x")
        
        with c2:
            fig_rules = px.bar(top_rules, x='lift', y='B', orientation='h',
                               title=f"與 '{sel_a}' 的關聯度排行",
                               labels={'lift': '關聯強度 (Lift)', 'B': '建議加購品項'},
                               color='lift', color_continuous_scale='GnBu')
            st.plotly_chart(fig_rules, use_container_width=True)
    else:
        st.info("💡 該區域目前交易關聯較為分散，建議查看 UK 市場以獲得更明顯的分析結果。")

with tab3:
    # 簡單乾淨的 KPI 呈現
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("訂單總量", f"{df_filtered['InvoiceNo'].nunique():,}")
    kpi2.metric("總銷售營收", f"£{df_filtered['TotalSum'].sum():,.0f}")
    kpi3.metric("活躍客戶數", f"{df_filtered['CustomerID'].nunique():,}")
    
    # 營收走勢
    df_filtered['Month'] = df_filtered['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = df_filtered.groupby('Month')['TotalSum'].sum().reset_index()
    fig_line = px.line(monthly_sales, x='Month', y='TotalSum', title="營收月走勢分析", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Optimized by Yi-Han")
