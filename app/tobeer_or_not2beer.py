import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 基礎設置與環境淨化 ---
warnings.filterwarnings("ignore")
#st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Retail Strategy Hub", page_icon="🍺", layout="wide")

# 強制隱藏 UI 雜訊
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- 1. 真實數據載入 ---
@st.cache_data
def load_grocery_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 建立商品名稱與顯示名稱的對照 (統一格式)
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 模擬商業價值欄位
    price_map = {'Bottled Beer': 18.5, 'Potato Chips': 5.0, 'Whole Milk': 3.5, 'Soda': 2.5}
    df['Price'] = df['Item'].map(price_map).fillna(np.random.uniform(3, 12))
    df['Profit'] = df['Price'] * 0.3
    return df

df_all = load_grocery_data()

# --- 2. 側邊欄 ---
st.sidebar.title("💎 策略控制中心")
lead_time = st.sidebar.select_slider("物流補貨天數 (Buffer)", options=[1, 3, 5, 7, 10, 14], value=5)

# --- 3. 主要 UI 分頁 ---
st.title("🛒 Supermarket Executive Center")
tab1, tab2, tab3 = st.tabs(["🔗 購物籃分析 (真實數據)", "📦 補貨緩衝與庫存", "👥 客戶價值分析"])

# --- Tab 1: 購物籃分析 (修復穩定版) ---
with tab1:
    st.subheader("🛍️ 真實交易關聯挖掘")
    
    # 為了防止 App 崩潰，我們限制分析最新 10,000 筆交易
    df_sample = df_all.head(10000)
    
    # 矩陣轉換：確保是「純布林矩陣」
    try:
        basket = df_sample.groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number')
        basket_bool = (basket > 0).astype(bool)
        
        with st.spinner('🔍 正在進行大數據運算...'):
            frequent_itemsets = apriori(basket_bool, min_support=0.005, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        if not rules.empty:
            rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # 搜尋框與預設值
            all_a = sorted(rules['A'].unique())
            default_item = "Bottled Beer" if "Bottled Beer" in all_a else all_a[0]
            
            sel_a = st.selectbox("🎯 選擇領頭商品 (Driver)：", all_a, index=all_a.index(default_item))
            
            top_rules = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).head(5)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("最強關聯搭配", top_rules['B'].iloc[0])
                st.metric("銷售提升 (Lift)", f"{top_rules['lift'].iloc[0]:.2f}x")
                st.success(f"**分析亮點：**\n\n購買 **{sel_a}** 的客戶，購買 **{top_rules['B'].iloc[0]}** 的機率是普通人的 {top_rules['lift'].iloc[0]:.1f} 倍。")
            with c2:
                fig_bar = px.bar(top_rules, x='lift', y='B', orientation='h', color='lift',
                                 labels={'B': '推薦搭配商品', 'lift': '關聯強度'},
                                 color_continuous_scale='Tealgrn')
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("目前數據量尚不足以產生顯著關聯規則。")
    except Exception as e:
        st.error(f"分析模組發生錯誤，請重新載入。({e})")

# --- Tab 2: 供應鏈 Buffer ---
with tab2:
    st.subheader("📦 補貨緩衝建議 (Supply Chain Buffer)")
    # 計算安全庫存邏輯
    inv = df_all.groupby('Item').agg({'Member_number': 'count'})
    inv.columns = ['Total_Sales']
    inv['Avg_Daily'] = inv['Total_Sales'] / df_all['Date'].nunique()
    inv['Volatility'] = inv['Avg_Daily'] * 0.4
    
    # 補貨 Buffer 公式 (95% 服務水準)
    inv['Buffer'] = (1.65 * inv['Volatility'] * np.sqrt(lead_time)).round(1)
    
    # 觀察關鍵品項
    target_items = ['Bottled Beer', 'Potato Chips', 'Whole Milk', 'Soda']
    display_inv = inv[inv.index.isin(target_items)].reset_index()
    
    fig_buffer = px.bar(display_inv, x='Item', y=['Avg_Daily', 'Buffer'],
                        title=f"前置時間 {lead_time} 天下的安全庫存組成",
                        barmode='group', color_discrete_sequence=['#457b9d', '#e63946'])
    st.plotly_chart(fig_buffer, use_container_width=True)

# --- Tab 3: RFM 客戶分析 ---
with tab3:
    st.subheader("👥 客戶貢獻分佈")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Member_number': 'count',
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Monetary",
                         hover_name=rfm.index, title="RFM 客戶雷達分佈",
                         color_continuous_scale='Viridis')
    st.plotly_chart(fig_rfm, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Yi-Han's Strategic Retail Hub")
