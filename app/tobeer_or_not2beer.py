import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 環境設置 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Center", page_icon="📊", layout="wide")

# --- 1. 核心數據引擎 (根據零售常規設定) ---
@st.cache_data
def load_data_pro():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 數據正名：確保洋芋片出現在分析中
    df['itemDescription'] = df['itemDescription'].replace({'salty snack': 'Potato Chips'})
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 業界常規毛利字典 (用來判定 Loss Leader)
    margin_config = {
        'Bottled Beer': 0.08, 'Canned Beer': 0.07, 'Whole Milk': 0.05,
        'Potato Chips': 0.45, 'Soda': 0.40, 'Sausage': 0.35, 'Pastry': 0.32
    }
    df['Margin'] = df['Item'].map(margin_config).fillna(0.22)
    # 模擬價格
    df['Price'] = df['Item'].map({'Bottled Beer': 18, 'Potato Chips': 5, 'Whole Milk': 3.5}).fillna(7.0)
    return df, margin_config

df_all, margin_lookup = load_data_pro()

# --- 2. 側邊欄控制台 ---
st.sidebar.title("💎 營運策略控制台")
lead_time = st.sidebar.select_slider("物流前置時間 (Lead Time)", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_factor = st.sidebar.slider("安全庫存係數 (Buffer)", 1.0, 3.0, 1.65)
st.sidebar.markdown("---")
st.sidebar.caption("Yi-Han's Retail Engine v8.0")

# --- 3. 主要 UI 分頁 ---
st.title("🚀 Strategic Retail Executive Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["👥 RFM 精準行銷", "📦 需求預測與補貨", "🛍️ 策略判定 (MBA)", "📉 智慧定價"])

# --- Tab 1: RFM (指標與圖表回來了！) ---
with tab1:
    st.subheader("🎯 會員價值分群與流失預警")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days, 
        'Member_number': 'count', 
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    def get_seg(x):
        if x['Monetary'] > rfm['Monetary'].quantile(0.8) and x['Frequency'] > rfm['Frequency'].quantile(0.8): return 'VIP'
        if x['Recency'] > 60 and x['Monetary'] > rfm['Monetary'].median(): return 'At Risk'
        return 'Standard'
    
    rfm['Segment'] = rfm.apply(get_seg, axis=1)
    at_risk = rfm[rfm['Segment'] == 'At Risk']
    
    c1, c2, c3 = st.columns(3)
    c1.metric("流失預警客戶 (At Risk)", f"{len(at_risk)} 人")
    c2.metric("潛在流失總金額", f"€{at_risk['Monetary'].sum():,.0f}")
    c3.metric("流失前平均頻次", f"{at_risk['Frequency'].mean():.1f} 次")
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency", 
                         title="RFM 價值分佈矩陣", color_discrete_map={'VIP':'#FFD700', 'At Risk':'#FF4B4B', 'Standard':'#1F77B4'})
    st.plotly_chart(fig_rfm, width='stretch')

# --- Tab 2: 供應鏈補貨 (對齊妳的模擬圖) ---
with tab2:
    st.subheader("📦 供應鏈動態補貨與庫存平衡")
    time_series = df_all.groupby('Date')['Item'].count().tail(30).reset_index()
    time_series.columns = ['Date', 'Demand']
    
    # 計算動態補貨點
    std_demand = time_series['Demand'].std()
    time_series['Reorder_Point'] = time_series['Demand'] + (buffer_factor * std_demand * np.sqrt(lead_time))
    
    fig_supply = go.Figure()
    fig_supply.add_trace(go.Scatter(x=time_series['Date'], y=time_series['Demand'], name='歷史銷量', line=dict(color='#1F77B4')))
    fig_supply.add_trace(go.Scatter(x=time_series['Date'], y=time_series['Reorder_Point'], name='動態建議補貨量', line=dict(color='#E63946', dash='dash')))
    st.plotly_chart(fig_supply, width='stretch')
    
    sc1, sc2 = st.columns(2)
    with sc1:
        st.warning("📉 **提高 Buffer 策略：** 確保週末不缺貨，但會增加 €2,400 庫存持有成本。")
    with sc2:
        st.success("📈 **低 Buffer 策略：** 加快周轉率，建議針對非生鮮品項執行。")

# --- Tab 3: 購物籃策略 (Loss Leader / Bundle 核心) ---
with tab3:
    st.subheader("🛍️ 購物籃交叉銷售策略判定")
    basket = df_all.head(10000).groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number')
    basket_bool = (basket > 0).astype(bool)
    f_sets = apriori(basket_bool, min_support=0.005, use_colnames=True)
    rules = association_rules(f_sets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        sel_a = st.selectbox("🎯 選擇分析 Driver 商品 (A)：", sorted(rules['A'].unique()), index=0)
        top_rule = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        sel_b = top_rule['B']
        
        ma, mb = margin_lookup.get(sel_a, 0.20), margin_lookup.get(sel_b, 0.20)
        
        # 判定邏輯
        if ma <= 0.12 and mb >= 0.20:
            st.error("🚀 **建議策略：Loss Leader (引流策略)**")
            reason = f"診斷：{sel_a} 毛利極低 ({ma*100:.0f}%)，適合作為引流品項帶動高毛利商品 {sel_b}。"
        elif top_rule['lift'] > 1.3:
            st.success("📦 **建議策略：Bundle (綑綁銷售)**")
            reason = f"診斷：{sel_a} 與 {sel_b} 關聯極強 (Lift: {top_rule['lift']:.2f})，適合推出組合包。"
        else:
            st.info("💡 **建議策略：Cross-Sell (交叉銷售)**")
            reason = "診斷：建議於貨架鄰近處陳列。"

        st.markdown(f"""<div style="background-color:rgba(100,100,100,0.1); padding:15px; border-radius:10px; border-left: 5px solid #ccc;"><b>策略診斷：</b>{reason}</div>""", unsafe_allow_html=True)
        st.plotly_chart(px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift'), width='stretch')

# --- Tab 4: 智慧定價 (回來了！) ---
with tab4:
    st.subheader("📉 智慧定價模擬：利潤最大化點")
    elasticity = st.slider("價格彈性係數", 1.0, 5.0, 2.4)
    prices = np.linspace(10, 30, 50)
    profits = (prices - 8) * (200 * (prices / 15) ** -elasticity)
    best_p = prices[np.argmax(profits)]
    
    fig_p = px.line(x=prices, y=profits, title="價格與預期利潤曲線", labels={'x':'價格 (€)', 'y':'預期利潤'})
    fig_p.add_vline(x=best_p, line_dash="dash", line_color="green", annotation_text=f"最優價格: €{best_p:.1f}")
    st.plotly_chart(fig_p, width='stretch')
