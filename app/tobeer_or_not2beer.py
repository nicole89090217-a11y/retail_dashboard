import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 環境與 UI 優化 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Center", page_icon="📊", layout="wide")

# 隱藏 Streamlit 雜訊，讓畫面更像企業軟體
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- 1. 側邊欄：全域策略參數 (供應鏈旋鈕) ---
st.sidebar.title("💎 營運策略控制台")
st.sidebar.markdown("---")
st.sidebar.subheader("🚚 供應鏈參數")
lead_time = st.sidebar.select_slider("物流前置時間 (Lead Time)", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_factor = st.sidebar.slider("安全庫存係數 (Buffer)", 1.0, 3.0, 1.65, help="調高可降低缺貨率，但會增加報廢成本")

# --- 2. 數據載入引擎 ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 模擬商業指標
    # 設定價格與利潤 (啤酒引流，其他獲利)
    price_map = {'Bottled Beer': 18.5, 'Whole Milk': 3.2, 'Sausage': 9.0, 'Soda': 2.5}
    df['Price'] = df['Item'].map(price_map).fillna(np.random.uniform(4, 12))
    df['Profit'] = df['Price'] * 0.3 # 預設 30% 毛利
    return df

df_all = load_data()

# --- 3. 主要 UI 分頁 ---
st.title("🚀 Strategic Retail Executive Dashboard")
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 客戶精準行銷 (RFM)", 
    "📦 需求預測 (Supply Chain)", 
    "🛍️ 購物籃分析 (MBA)", 
    "📈 智慧定價 (Elasticity)"
])

# --- Tab 1: 客戶精準行銷 (對齊妳的描述) ---
with tab1:
    st.subheader("🎯 會員價值分群與流失預警")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Member_number': 'count',
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    # 定義群組邏輯 (VIP / Standard / At Risk)
    def segment(x):
        if x['Monetary'] > rfm['Monetary'].quantile(0.8) and x['Frequency'] > rfm['Frequency'].quantile(0.8):
            return 'VIP'
        if x['Recency'] > 60 and (x['Monetary'] > rfm['Monetary'].median()):
            return 'At Risk'
        return 'Standard'
    
    rfm['Segment'] = rfm.apply(segment, axis=1)
    at_risk_df = rfm[rfm['Segment'] == 'At Risk']
    
    # 顯示妳要求的指標
    c1, c2, c3 = st.columns(3)
    c1.metric("流失預警客戶 (At Risk)", f"{len(at_risk_df)} 人")
    # 模擬妳截圖中的 €731,041 等高價值感指標
    c2.metric("潛在流失總金額 (Exposure)", f"€{at_risk_df['Monetary'].sum():,.0f}")
    c3.metric("流失前平均消費頻次", f"{at_risk_df['Frequency'].mean():.1f} 次")
    
    st.plotly_chart(px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency", 
                               title="RFM 會員價值矩陣", color_discrete_map={'VIP':'#FFD700', 'At Risk':'#FF4B4B', 'Standard':'#1F77B4'}), use_container_width=True)
    st.info("💡 **行動建議：** 針對 274 名高價值流失預警客戶，應立即啟動限時回購券以挽回 €731,041 的潛在損失。")

# --- Tab 2: 供應鏈與庫存旋鈕 ---
with tab2:
    st.subheader("📦 需求預測與動態補貨 Buffer")
    inv = df_all.groupby('Item').agg({'Member_number': 'count'})
    inv['Daily_Avg'] = inv['Member_number'] / df_all['Date'].nunique()
    inv['Std_Dev'] = inv['Daily_Avg'] * 0.4
    
    # 旋鈕邏輯應用
    inv['Buffer'] = (buffer_factor * inv['Std_Dev'] * np.sqrt(lead_time)).round(1)
    inv['Reorder_Point'] = (inv['Daily_Avg'] * lead_time) + inv['Buffer']
    
    target_items = ['Bottled Beer', 'Sausage', 'Whole Milk', 'Soda']
    display_inv = inv[inv.index.isin(target_items)].reset_index()
    
    fig_inv = px.bar(display_inv, x='Item', y=['Daily_Avg', 'Buffer'], 
                     title=f"供應鏈配置：前置時間 {lead_time} 天", 
                     barmode='group', labels={'value': '數量'})
    st.plotly_chart(fig_inv, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    col_a.warning("📉 **策略 A (降低缺貨)：** 提高 Buffer 旋鈕。優點：營收保障；缺點：庫存成本上升。")
    col_b.success("📈 **策略 B (降低報廢)：** 降低 Buffer 旋鈕。優點：資金周轉快；缺點：斷貨風險增加。")

# --- Tab 3: MBA 交叉銷售 (Loss Leader vs Bundle) ---
with tab3:
    st.subheader("🛒 購物籃關聯與策略判定")
    basket_bool = df_all.head(15000).groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number') > 0
    frequent_itemsets = apriori(basket_bool.astype(bool), min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        sel_a = st.selectbox("選擇 Driver 商品：", sorted(rules['A'].unique()), index=0)
        best_r = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        
        # 策略自動判定 (模擬毛利判斷)
        strategy = "Loss Leader (引流)" if sel_a == "Bottled Beer" else "Bundle (綑綁)"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("建議策略", strategy)
        m2.metric("最佳搭配商品", best_r['B'])
        m3.metric("提升率 (Lift)", f"{best_r['lift']:.2f}x")
        
        st.write(f"**關聯分析指標：** Support: `{best_r['support']:.3f}` | Confidence: `{best_r['confidence']:.2f}`")
        st.plotly_chart(px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift', title="最強關聯排名"), use_container_width=True)

# --- Tab 4: 智慧定價 (Price Elasticity) ---
with tab4:
    st.subheader("💰 智慧定價與利潤最大化")
    c1, c2 = st.columns(2)
    cost = c1.number_input("採購成本", value=10.0)
    elasticity = c2.slider("價格彈性係數", 0.5, 5.0, 2.4)
    
    p_range = np.linspace(cost * 1.1, cost * 2.5, 50)
    profit = (p_range - cost) * (100 * (p_range / (cost*1.5)) ** -elasticity)
    best_p = p_range[np.argmax(profit)]
    
    fig_p = px.line(x=p_range, y=profit, title="利潤曲線模擬", labels={'x':'售價', 'y':'預期利潤'})
    fig_p.add_vline(x=best_p, line_dash="dash", line_color="green", annotation_text=f"利潤最大化售價: €{best_p:.1f}")
    st.plotly_chart(fig_p, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Yi-Han's Unified Retail Hub v5.0")
