import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Lidl 門市中控台", layout="wide")
st.title("🛒 Lidl Store Operation Center")
st.markdown("### 全通路零售決策系統 (Omnichannel Retail Decision System)")
st.info("整合 **CRM (客群)**、**Supply Chain (庫存)** 與 **Profit Strategy (獲利)** 的三合一戰情室。")

# 建立三個分頁
tab1, tab2, tab3 = st.tabs(["👥 客戶精準行銷 (RFM)", "📦 智慧庫存預測 (Inventory)", "🧺 購物籃獲利策略 (Basket)"])

# ==========================================
# 分頁 1: CRM 設定 (RFM)
# ==========================================
with tab1:
    st.header("👥 客戶分群與挽回策略")
    
    # 模擬數據
    @st.cache_data
    def load_rfm_data():
        np.random.seed(42)
        data = pd.DataFrame({
            'CustomerID': range(1000, 2000),
            'Recency': np.random.randint(1, 100, 1000),
            'Monetary': np.random.randint(50, 5000, 1000)
        })
        return data

    df_rfm = load_rfm_data()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("參數設定")
        vip_threshold = st.slider("🏆 VIP 金額門檻 (€)", 1000, 5000, 3000)
        risk_threshold = st.slider("⚠️ 流失天數門檻 (Days)", 30, 120, 60)
    
    with col2:
        # 動態分群
        def segment(row):
            if row['Monetary'] >= vip_threshold: return 'VIP'
            if row['Recency'] >= risk_threshold: return 'At Risk'
            return 'Standard'
        
        df_rfm['Segment'] = df_rfm.apply(segment, axis=1)
        
        # 顯示 KPI
        risk_users = df_rfm[df_rfm['Segment']=='At Risk']
        risk_value = risk_users['Monetary'].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("VIP 人數", f"{len(df_rfm[df_rfm['Segment']=='VIP'])} 人")
        m2.metric("流失預警人數", f"{len(risk_users)} 人", delta="-需挽回", delta_color="inverse")
        m3.metric("潛在流失金額", f"€{risk_value:,.0f}")
        
        # 畫圖
        fig = px.scatter(df_rfm, x='Recency', y='Monetary', color='Segment', 
                         title="RFM 客戶價值分佈圖", color_discrete_map={'VIP':'green', 'At Risk':'red', 'Standard':'blue'})
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 分頁 2: 供應鏈設定 (Inventory)
# ==========================================
with tab2:
    st.header("📦 Prophet 動態庫存調節")
    
    # 模擬預測數據
    dates = pd.date_range(start='2026-01-01', periods=30)
    base_demand = 100
    demand = [int(base_demand * (1.4 if d.dayofweek >= 5 else 1.0) + np.random.randint(-10, 10)) for d in dates]
    df_inv = pd.DataFrame({'Date': dates, 'Forecast': demand})
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("供應鏈參數")
        safety_buffer = st.slider("🛡️ 安全庫存係數 (%)", 0, 50, 10)
        unit_cost = st.number_input("進貨成本 (€)", 0.5)
        
    with col2:
        df_inv['Order_Qty'] = df_inv['Forecast'] * (1 + safety_buffer/100)
        waste_cost = ((df_inv['Order_Qty'] - df_inv['Forecast']) * unit_cost).sum()
        
        m1, m2 = st.columns(2)
        m1.metric("建議總訂貨量", f"{int(df_inv['Order_Qty'].sum())}", delta=f"+{safety_buffer}% Buffer")
        m2.metric("預估報廢成本 (保險費)", f"€{waste_cost:,.0f}", delta_color="inverse")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_inv['Date'], y=df_inv['Forecast'], name='AI 預測需求'))
        fig.add_trace(go.Scatter(x=df_inv['Date'], y=df_inv['Order_Qty'], name='建議訂貨量', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 分頁 3: 購物籃獲利策略 (Basket Analysis) - NEW!
# ==========================================
with tab3:
    st.header("🧺 購物籃交叉銷售策略 (Cross-Selling Strategy)")
    st.markdown("利用 **關聯規則 (Association Rules)** 找出「帶路雞」，以低毛利商品帶動高毛利營收。")
    
    col_ui, col_kpi = st.columns([1, 2])
    
    with col_ui:
        st.subheader("🔍 選擇帶路雞商品 (Driver)")
        
        # 這裡模擬你算出來的規則
        rules_db = {
            'Beer 🍺': {
                'target': 'Chips 🥔',
                'lift': 5.0,
                'profit_driver': 0.10,  # 啤酒利潤
                'profit_target': 0.70,  # 洋芋片利潤
                'desc': '週末狂歡組合'
            },
            'Milk 🥛': {
                'target': 'Bread 🍞',
                'lift': 1.8,
                'profit_driver': 0.05,
                'profit_target': 0.35,
                'desc': '每日早餐剛需'
            },
            'Diapers 👶': {
                'target': 'Beer 🍺',
                'lift': 3.5,
                'profit_driver': 2.00,
                'profit_target': 0.50,
                'desc': '新手爸爸組合'
            }
        }
        
        selected_item = st.selectbox("請選擇促銷商品：", list(rules_db.keys()))
        rule = rules_db[selected_item]
        
    with col_kpi:
        # 計算數據
        total_profit = rule['profit_driver'] + rule['profit_target']
        profit_boost = (rule['profit_target'] / rule['profit_driver']) * 100
        
        st.subheader(f"📊 分析結果：{rule['desc']}")
        
        # 顯示 3 個大指標
        k1, k2, k3 = st.columns(3)
        k1.metric("關聯商品 (Target)", rule['target'])
        k2.metric("提升度 (Lift)", f"{rule['lift']}x", delta="極強關聯")
        k3.metric("組合總利潤", f"€{total_profit:.2f}", delta=f"+{profit_boost:.0f}% vs 單賣")
        
        # 畫一個簡單的利潤構成圖
        profit_data = pd.DataFrame({
            'Product': ['Driver (帶路雞)', 'Target (被帶動)'],
            'Profit': [rule['profit_driver'], rule['profit_target']],
            'Color': ['#bdc3c7', '#27ae60'] # 灰色是低毛利，綠色是高毛利
        })
        fig_bar = px.bar(profit_data, x='Product', y='Profit', color='Product', 
                         title="單品利潤貢獻比較 (Profit Contribution)",
                         color_discrete_sequence=['#7f8c8d', '#2ecc71'])
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- 自動生成策略建議 (Auto-Strategy) ---
    st.markdown("---")
    st.subheader("💡策略建議 (Actionable Insight)")
    
    if rule['profit_driver'] < rule['profit_target']:
        strategy_text = f"""
        **建議策略：Loss Leader Strategy (帶路雞策略)**
        * **洞察：** {selected_item} 的利潤極低 (€{rule['profit_driver']})，但它是 {rule['target']} 的強力流量入口 (Lift: {rule['lift']})。
        * **行動：** 建議對 {selected_item} 進行 **降價促銷** 甚至成本價販售，吸引客流。
        * **預期結果：** 雖然 {selected_item} 不賺錢，但每賣出一個，有高機率連帶銷售高毛利的 {rule['target']} (€{rule['profit_target']})，使整體購物籃獲利最大化。
        """
        st.success(strategy_text)
    else:
        strategy_text = f"""
        **建議策略：Bundle Strategy (強強聯手)**
        * **洞察：** 兩者皆為高利潤商品，且關聯度高。
        * **行動：** 推出「組合包」或是將兩者陳列在一起。
        """
        st.info(strategy_text)
with tab4: 
    # 假設這是新分頁
    st.header("💰 價格彈性與獲利模擬 (Price Elasticity)")
    st.markdown("模擬 **價格變動** 對 **需求量** 的影響，尋找獲利最大化的甜蜜點。")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("參數設定")
        base_price = st.number_input("目前售價 (€)", 1.0, 100.0, 10.0)
        base_cost = st.number_input("商品成本 (€)", 0.5, 50.0, 6.0)
        base_demand = st.number_input("目前日銷量", 10, 1000, 100)
        
        # 彈性係數：-2.0 代表漲價 1%，銷量掉 2% (對價格敏感)
        elasticity = st.slider("價格彈性係數 (Elasticity)", -3.0, -0.1, -1.5, step=0.1,
                               help="絕對值越大，代表客戶對價格越敏感（一漲價就跑）。")

    with col2:
        # 模擬價格從 -20% 到 +20% 的變化
        price_change_pct = np.linspace(-0.2, 0.2, 50)
        sim_prices = base_price * (1 + price_change_pct)
        
        # 需求公式：Q_new = Q_old * (1 + Elasticity * %Price_Change)
        sim_demand = base_demand * (1 + elasticity * price_change_pct)
        
        # 獲利公式：Profit = (Price - Cost) * Demand
        sim_profit = (sim_prices - base_cost) * sim_demand
        
        # 找出最大獲利點
        max_profit_idx = np.argmax(sim_profit)
        best_price = sim_prices[max_profit_idx]
        best_profit = sim_profit[max_profit_idx]
        
        st.metric("建議最佳售價", f"€{best_price:.2f}", delta=f"{(best_price-base_price)/base_price:.1%}")
        st.metric("預估最大獲利", f"€{best_profit:.1f}")

        # 畫圖
        df_sim = pd.DataFrame({
            'Price': sim_prices,
            'Profit': sim_profit,
            'Demand': sim_demand
        })
        
        fig = px.line(df_sim, x='Price', y=['Profit', 'Demand'], markers=True, 
                      title="價格 vs. 獲利/需求 敏感度分析")
        fig.add_vline(x=best_price, line_dash="dash", line_color="green", annotation_text="最佳定價")
        st.plotly_chart(fig, use_container_width=True)
with tab5: 
    # 假設這是另一個新分頁
    st.header("🗺️ 客戶地理分佈 (Geospatial Insights)")
    st.markdown("分析 Heilbronn 地區的客戶密度，優化 **門市選址** 與 **物流配送**。")

    # 模擬數據：生成 Heilbronn 附近的座標 (緯度 49.14, 經度 9.21)
    @st.cache_data
    def load_geo_data():
        n_points = 500
        # 在 Heilbronn 中心點附近隨機生成
        lat = np.random.normal(49.1427, 0.02, n_points)
        lon = np.random.normal(9.2109, 0.02, n_points)
        return pd.DataFrame({'lat': lat, 'lon': lon})

    df_map = load_geo_data()

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 簡單的地圖
        st.map(df_map)
    
    with col2:
        st.info("💡 **商業洞察：**")
        st.markdown("""
        * **熱區發現：** 客戶高度集中在市中心東北側。
        * **行動建議：** 建議在該區域增設 **Lidl Connect 取貨點** 或做為 **生鮮快送 (Quick Commerce)** 的前置倉 (Dark Store)。
        """)
