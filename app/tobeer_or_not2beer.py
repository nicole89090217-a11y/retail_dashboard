import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 淨化環境：隱藏所有不必要的警告 ---
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- 1. 網頁配置 ---
st.set_page_config(page_title="Retail Insights", page_icon="🍺", layout="wide")

# 強制隱藏 Streamlit 預設的右上角選單，讓畫面更乾淨
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 真實數據引擎 (Groceries Dataset) ---
@st.cache_data
def load_and_clean_data():
    # 使用包含「啤酒與洋芋片」的真實超市交易數據
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 模擬商業欄位 (為了 RFM 與 營收分析)
    price_dict = {'bottled beer': 18.5, 'potato chips': 4.5, 'whole milk': 3.2, 'soda': 2.0}
    df['Price'] = df['itemDescription'].map(price_dict).fillna(np.random.uniform(3, 12))
    df['Profit'] = df['Price'] * 0.35
    
    # 統一縮短品名，避免圖表雜亂
    df['Item'] = df['itemDescription'].str[:18].str.title()
    return df

df_all = load_and_clean_data()

# --- 3. 側邊欄設計 ---
st.sidebar.title("💎 營運決策中心")
st.sidebar.markdown("---")
lead_time = st.sidebar.select_slider("物流前置時間 (天)", options=[1, 3, 5, 7, 10, 14], value=5)

# --- 4. 主要介面 ---
st.title("🛒 Supermarket Executive Dashboard")

tab1, tab2, tab3 = st.tabs(["🔗 關聯分析 (Real MBA)", "📦 補貨緩衝 (Buffer)", "👥 客戶價值 (RFM)"])

# --- Tab 1: 真實購物籃關聯 ---
with tab1:
    st.subheader("哪些商品具有強烈的連帶購買關係？")
    
    # 優化運算量：取前 10,000 筆確保不當機
    df_sample = df_all.head(10000)
    
    # 建立矩陣並「強制轉換為布林值」 (消除妳看到的警告)
    basket = df_sample.groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number')
    basket_sets = (basket > 0).astype(bool) # 關鍵修正
    
    with st.spinner('🔍 正在掃描真實交易關聯...'):
        frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # 讓用戶搜尋啤酒或任何商品
        search_q = st.selectbox("🎯 選擇或搜尋領頭商品：", sorted(rules['A'].unique()), 
                               index=list(sorted(rules['A'].unique())).index('Bottled Beer') if 'Bottled Beer' in rules['A'].unique() else 0)
        
        top_rules = rules[rules['A'] == search_q].sort_values('lift', ascending=False).head(5)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("最強搭配商品", top_rules['B'].iloc[0])
            st.metric("銷售提升率 (Lift)", f"{top_rules['lift'].iloc[0]:.2f}x")
            st.info(f"**戰術建議：**\n\n購買 **{search_q}** 的人，買 **{top_rules['B'].iloc[0]}** 的機率是普通人的 {top_rules['lift'].iloc[0]:.1f} 倍。")
        with c2:
            fig_rules = px.bar(top_rules, x='lift', y='B', orientation='h', color='lift',
                               labels={'B': '推薦商品', 'lift': '關聯強度 (Lift)'},
                               color_continuous_scale='Mint')
            fig_rules.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_rules, use_container_width=True)
    else:
        st.error("暫無顯著關聯，請調整 Support 門檻。")

# --- Tab 2: 供應鏈 Buffer ---
with tab2:
    st.subheader("安全庫存緩衝建議")
    inv = df_all.groupby('Item').agg({'Member_number': 'count'})
    inv.columns = ['Total_Sales']
    inv['Avg_Daily'] = inv['Total_Sales'] / df_all['Date'].nunique()
    inv['Volatility'] = inv['Avg_Daily'] * 0.4 # 模擬變動係數
    
    # Buffer 計算 (95% 服務水準)
    inv['Buffer'] = (1.65 * inv['Volatility'] * np.sqrt(lead_time)).round(1)
    inv['Reorder_Point'] = (inv['Avg_Daily'] * lead_time) + inv['Buffer']
    
    # 專注看啤酒與洋芋片
    target_items = ['Bottled Beer', 'Potato Chips', 'Soda', 'Whole Milk']
    display_inv = inv[inv.index.isin(target_items)].reset_index()
    
    fig_buffer = px.bar(display_inv, x='Item', y=['Avg_Daily', 'Buffer'],
                        title=f"前置時間 {lead_time} 天下的安全庫存組成",
                        labels={'value': '數量', 'variable': '庫存類型'},
                        barmode='group', color_discrete_sequence=['#AEC6CF', '#FFB347'])
    st.plotly_chart(fig_buffer, use_container_width=True)

# --- Tab 3: 會員價值 ---
with tab3:
    st.subheader("RFM 會員價值分佈")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Member_number': 'count',
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Monetary",
                         hover_name=rfm.index, title="客戶分佈圖 (顏色越深貢獻越高)",
                         color_continuous_scale='Purp')
    st.plotly_chart(fig_rfm, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard engineered by Yi-Han | Data Analytics Pro")
