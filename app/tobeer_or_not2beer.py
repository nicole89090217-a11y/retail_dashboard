import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 基礎配置與環境淨化 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Dash", page_icon="🍺", layout="wide")

# --- 1. 數據載入與商業維度擴增 ---
@st.cache_data
def load_and_refine_data():
    # 使用真實超市交易大數據
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 【正名運動】將數據中的雜項正名為妳要的 Potato Chips
    df['itemDescription'] = df['itemDescription'].replace({
        'salty snack': 'Potato Chips',
        'bottled beer': 'Bottled Beer',
        'canned beer': 'Canned Beer'
    })
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 模擬利潤與成本 (用於策略判定)
    price_map = {'Bottled Beer': 18, 'Potato Chips': 5, 'Whole Milk': 3, 'Sausage': 8}
    df['Price'] = df['Item'].map(price_map).fillna(np.random.uniform(4, 10))
    # 模擬毛利：啤酒低(0.12)，洋芋片高(0.45)
    margin_map = {'Bottled Beer': 0.12, 'Potato Chips': 0.45, 'Whole Milk': 0.05}
    df['Margin'] = df['Item'].map(margin_map).fillna(0.20)
    df['TotalSum'] = df['Price'] # 簡化計算
    return df

df_all = load_and_refine_data()

# --- 2. 側邊欄：全域變數定義 ---
st.sidebar.title("💎 策略控制中心")
lead_time = st.sidebar.select_slider("物流補貨天數 (Lead Time)", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_val = st.sidebar.slider("調整補貨 Buffer 旋鈕", 1.0, 3.0, 1.65)

# --- 3. 主要 UI ---
st.title("🛒 Strategic Retail Operation Center")

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 客戶精準行銷 (RFM)", 
    "🚚 需求預測 (Supply Chain)", 
    "🛍️ 交叉銷售 (MBA)", 
    "📉 智慧定價 (Elasticity)"
])

# --- 模組 1: RFM 客戶分析 ---
with tab1:
    st.subheader("🎯 客戶分群與流失預警")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Member_number': 'count',
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    # 分群邏輯
    rfm['Segment'] = 'Standard'
    rfm.loc[(rfm['Monetary'] > rfm['Monetary'].quantile(0.8)) & (rfm['Frequency'] > rfm['Frequency'].quantile(0.8)), 'Segment'] = 'VIP'
    rfm.loc[(rfm['Recency'] > 60) & (rfm['Monetary'] > rfm['Monetary'].median()), 'Segment'] = 'At Risk'
    
    at_risk_df = rfm[rfm['Segment'] == 'At Risk']
    c1, c2, c3 = st.columns(3)
    c1.metric("流失預警客戶數", f"{len(at_risk_df)} 人")
    c2.metric("潛在流失總金額", f"€{at_risk_df['Monetary'].sum():,.0f}")
    c3.metric("流失前平均消費次數", f"{at_risk_df['Frequency'].mean():.1f} 次")
    
    st.plotly_chart(px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency", 
                               title="客戶價值分佈圖", color_discrete_map={'VIP':'gold', 'At Risk':'red', 'Standard':'blue'}), use_container_width=True)

# --- 模組 2: Supply Chain ---
with tab2:
    st.subheader("📦 需求預測與補貨 Buffer")
    inv = df_all.groupby('Item').agg({'Member_number': 'count'})
    inv['Daily_Avg'] = inv['Member_number'] / df_all['Date'].nunique()
    inv['Std_Dev'] = inv['Daily_Avg'] * 0.5 
    inv['Safety_Stock'] = (buffer_val * inv['Std_Dev'] * np.sqrt(lead_time)).round(1)
    
    targets = ['Bottled Beer', 'Potato Chips', 'Sausage', 'Whole Milk']
    existing = [i for i in targets if i in inv.index]
    if existing:
        fig_inv = px.bar(inv.loc[existing].reset_index(), x='Item', y=['Daily_Avg', 'Safety_Stock'], 
                         title="預期需求 vs. 安全緩衝 (Buffer)", barmode='group')
        st.plotly_chart(fig_inv, use_container_width=True)

# --- 模組 3: MBA (Beer & Chips) ---
with tab3:
    st.subheader("🛒 購物籃交叉銷售策略")
    # 建立矩陣 (取部分數據確保速度)
    basket_bool = df_all.head(15000).groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number') > 0
    frequent_itemsets = apriori(basket_bool.astype(bool), min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        sel_a = st.selectbox("選擇 Driver 商品：", sorted(rules['A'].unique()), 
                             index=list(sorted(rules['A'].unique())).index('Bottled Beer') if 'Bottled Beer' in rules['A'].unique() else 0)
        
        rule = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        strategy = "Loss Leader (引流)" if sel_a == "Bottled Beer" else "Bundle (綑綁)"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("建議策略", strategy)
        m2.metric("推薦搭配", rule['B'])
        m3.metric("提升率 (Lift)", f"{rule['lift']:.2f}x")
        
        fig_mba = px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift', title="最強關聯商品排行")
        st.plotly_chart(fig_mba, use_container_width=True)

# --- 模組 4: Price Elasticity ---
with tab4:
    st.subheader("💰 智慧定價利潤模擬")
    e_val = st.slider("產品價格彈性係數", 0.5, 5.0, 2.4)
    cost = 10.0
    p_range = np.linspace(11, 30, 50)
    profit = (p_range - cost) * (100 * (p_range / 15) ** -e_val)
    best_p = p_range[np.argmax(profit)]
    
    fig_p = px.line(x=p_range, y=profit, title="利潤最大化曲線", labels={'x':'售價', 'y':'預期利潤'})
    fig_p.add_vline(x=best_p, line_dash="dash", line_color="green", annotation_text=f"最佳售價: €{best_p:.1f}")
    st.plotly_chart(fig_p, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Yi-Han's Strategic Retail Engine v4.0")
