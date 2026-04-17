import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 基礎配置與環境淨化 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Retail Strategy Hub", page_icon="📊", layout="wide")

# 強制隱藏 Streamlit 預設雜訊，讓介面像企業級軟體
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- 1. 側邊欄：先定義全域變數 (防止 NameError) ---
st.sidebar.title("💎 策略控制中心")
lead_time = st.sidebar.select_slider("物流補貨前置天數 (Lead Time Buffer)", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_factor = st.sidebar.slider("安全庫存係數 (Confidence Interval)", 1.0, 3.0, 1.65, help="1.65 對應 95% 不斷貨率")

# --- 2. 數據載入與商業維度擴增 ---
@st.cache_data
def load_and_refine_data():
    # 使用真實超市交易大數據 (Groceries Dataset)
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 【關鍵：數據標準化】確保 Beer & Chips 存在且名稱統一
    df['itemDescription'] = df['itemDescription'].replace({
        'salty snack': 'Potato Chips',
        'bottled beer': 'Bottled Beer',
        'canned beer': 'Canned Beer'
    })
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 模擬利潤、成本與價格 (用於展示價格彈性與策略)
    # 定義毛利：啤酒低毛利(引流)，洋芋片高毛利(獲利)
    price_map = {'Bottled Beer': 18.5, 'Potato Chips': 5.0, 'Whole Milk': 3.5, 'Sausage': 8.0}
    margin_map = {'Bottled Beer': 0.12, 'Potato Chips': 0.45, 'Whole Milk': 0.05, 'Sausage': 0.30}
    
    df['Price'] = df['Item'].map(price_map).fillna(np.random.uniform(4, 12))
    df['Margin'] = df['Item'].map(margin_map).fillna(0.20)
    df['Profit'] = df['Price'] * df['Margin']
    df['Cost'] = df['Price'] - df['Profit']
    
    return df

df_all = load_and_refine_data()

# --- 3. 主要 UI 分頁 ---
st.title("🛒 Strategic Retail Operation Center")
st.markdown("這份看板整合了 **客戶精準行銷**、**供應鏈韌性**、**交叉銷售** 與 **價格優化** 四大模組。")

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 客戶精準行銷 (RFM)", 
    "🚚 需求預測 (Supply Chain)", 
    "🛍️ 交叉銷售 (MBA)", 
    "📉 智慧定價 (Elasticity)"
])

# --- 模組 1: 客戶精準行銷 (RFM) ---
with tab1:
    st.subheader("🎯 客戶分群與流失預警分析")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Member_number': 'count',
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    # 分群邏輯
    m_high = rfm['Monetary'].quantile(0.7)
    f_high = rfm['Frequency'].quantile(0.7)
    r_high = 60 # 假設超過 60 天沒來即有流失風險
    
    def get_segment(x):
        if x['Monetary'] > m_high and x['Frequency'] > f_high: return 'VIP'
        if x['Recency'] > r_high and (x['Monetary'] > m_high or x['Frequency'] > f_high): return 'At Risk'
        return 'Standard'
    
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    at_risk = rfm[rfm['Segment'] == 'At Risk']
    
    c1, c2, c3 = st.columns(3)
    c1.metric("流失預警客戶數", f"{len(at_risk)} 人")
    c2.metric("潛在流失總金額", f"€{at_risk['Monetary'].sum():,.0f}")
    c3.metric("流失前平均消費", f"{at_risk['Frequency'].mean():.1f} 次")
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency",
                         title="RFM 客戶價值分佈矩陣", labels={'Recency': '最後一次消費天數'})
    st.plotly_chart(fig_rfm, use_container_width=True)
    st.info("💡 **決策建議：** 系統偵測到高價值流失風險，建議立即針對 At Risk 族群發送專屬回購優惠券。")

# --- 模組 2: 供應鏈與物流 (Supply Chain) ---
with tab2:
    st.subheader("📦 供應鏈補貨 Buffer 管理")
    inv = df_all.groupby('Item').agg({'Member_number': 'count'})
    inv.columns = ['Total_Sales']
    inv['Avg_Daily'] = inv['Total_Sales'] / df_all['Date'].nunique()
    inv['Std_Dev'] = inv['Avg_Daily'] * 0.4 # 模擬銷售波動
    
    # 使用側邊欄定義的 lead_time 和 buffer_factor
    inv['Safety_Buffer'] = (buffer_factor * inv['Std_Dev'] * np.sqrt(lead_time)).round(1)
    inv['Reorder_Point'] = (inv['Avg_Daily'] * lead_time) + inv['Safety_Buffer']
    
    # 防錯機制：確保觀察清單存在於 Index 中
    targets = ['Bottled Beer', 'Potato Chips', 'Sausage', 'Whole Milk']
    existing_targets = [i for i in targets if i in inv.index]
    
    if existing_targets:
        display_inv = inv.loc[existing_targets].reset_index()
        fig_inv = px.bar(display_inv, x='Item', y=['Avg_Daily', 'Safety_Buffer'],
                        title=f"前置時間 {lead_time} 天下的庫存需求配置",
                        labels={'value': '數量', 'variable': '庫存組成'},
                        barmode='group', color_discrete_sequence=['#457b9d', '#e63946'])
        st.plotly_chart(fig_inv, use_container_width=True)
    
    st.warning(f"當前 Buffer 旋鈕設定為 {buffer_factor}: 旨在平衡缺貨損失與庫存持有成本。")

# --- 模組 3: 購物籃交叉銷售 (MBA) ---
with tab3:
    st.subheader("🛒 購物籃關聯與策略判定")
    df_mba = df_all.head(15000) # 優化效能
    basket_bool = df_mba.groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number') > 0
    
    frequent_itemsets = apriori(basket_bool.astype(bool), min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        all_a = sorted(rules['A'].unique())
        # 預設選中 Bottled Beer
        def_idx = all_a.index("Bottled Beer") if "Bottled Beer" in all_a else 0
        sel_a = st.selectbox("選擇 Driver 商品：", all_a, index=def_idx)
        
        best_match = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        
        # 策略判定：根據模擬毛利，低於 15% 視為 Loss Leader
        strategy = "Loss Leader (帶路雞策略)" if sel_a == "Bottled Beer" else "Bundle (綑綁銷售策略)"
        
        col1, col2 = st.columns([1, 2])
        col1.metric("建議策略", strategy)
        col1.metric("關聯提升 (Lift)", f"{best_match['lift']:.2f}x")
        
        fig_mba = px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift', title="最強關聯排名")
        col2.plotly_chart(fig_mba, use_container_width=True)
    else:
        st.info("數據計算中，請切換至 United Kingdom 數據集。")

# --- 模組 4: 智慧定價模擬 (Elasticity) ---
with tab4:
    st.subheader("💰 價格彈性與利潤極大化")
    c_p1, c_p2 = st.columns(2)
    
    target_cost = c_p1.number_input("輸入採購成本 (Cost)", value=10.0)
    current_rsp = c_p1.number_input("當前建議售價 (RSP)", value=15.0)
    elasticity_val = c_p2.slider("產品價格彈性係數", 0.0, 5.0, 2.4, help=">1 代表高彈性，適合降價")
    
    # 利潤公式模擬
    price_range = np.linspace(target_cost * 1.1, current_rsp * 1.8, 50)
    sim_demand = 100 * (price_range / current_rsp) ** -elasticity_val
    sim_profit = (price_range - target_cost) * sim_demand
    
    max_p_idx = np.argmax(sim_profit)
    best_p = price_range[max_p_idx]
    
    fig_p = px.line(x=price_range, y=sim_profit, title="售價與預期總利潤模擬曲線",
                    labels={'x': '售價 (€)', 'y': '預期利潤'})
    fig_p.add_vline(x=best_p, line_dash="dash", line_color="green", annotation_text=f"最優定價: €{best_p:.1f}")
    st.plotly_chart(fig_p, use_container_width=True)
    st.success(f"🎯 **定價建議：** 針對彈性係數 {elasticity_val} 的商品，建議將售價設定在 **€{best_p:.1f}**。")

st.sidebar.markdown("---")
st.sidebar.caption("Yi-Han's Unified Retail Engine v3.0")
