import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 1) Page setup
# =========================
st.set_page_config(page_title="Retail Store 門市中控台", layout="wide")
st.title("🛒 Retail Store Operation Center")
st.markdown("### 全通路零售決策系統 (Omnichannel Retail Decision System)")
st.info("整合 **CRM (客群)**、**Supply Chain (庫存)** 與 **Profit Strategy (獲利)** 的三合一零售戰情室。")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 客戶精準行銷 (RFM)",
    "📦 需求預測與補貨 (Inventory)",
    "🧺 購物籃獲利策略 (Basket)",
    "💰 智慧定價模擬 (Pricing)",
    "🗺️ 客戶地理分佈 (Location)"
])

# =========================
# Tab 1: CRM (RFM)
# =========================
with tab1:
    st.header("👥 客戶分群與挽回策略 (RFM)")

    @st.cache_data
    def load_rfm_data(n=1000, seed=42):
        np.random.seed(seed)
        df = pd.DataFrame({
            "CustomerID": range(1000, 1000 + n),
            "Recency": np.random.randint(1, 120, n),          # days since last purchase
            "Frequency": np.random.randint(1, 25, n),         # purchase count
            "Monetary": np.random.randint(20, 6000, n)        # total spend (€)
        })
        return df

    df_rfm = load_rfm_data()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("參數設定")
        vip_m_threshold = st.slider("🏆 VIP 金額門檻 Monetary (€)", 500, 6000, 3000, step=100)
        vip_f_threshold = st.slider("🏆 VIP 次數門檻 Frequency", 1, 25, 10)
        risk_recency = st.slider("⚠️ 流失天數門檻 Recency (Days)", 15, 120, 60)
        risk_value_floor = st.slider("⚠️ 流失預警最低價值 (€)", 0, 6000, 800, step=100)

        with st.expander("Methodology & assumptions"):
            st.markdown(
                "- 這裡的 R/F/M 目前為**示範用模擬資料**。\n"
                "- VIP：Monetary 高 且 Frequency 高。\n"
                "- At Risk：Recency 高 且（Monetary 或 Frequency）不低，避免把低價值客戶誤判為需挽回對象。"
            )

    with col2:
        def segment(row):
            if (row["Monetary"] >= vip_m_threshold) and (row["Frequency"] >= vip_f_threshold):
                return "VIP"
            if (row["Recency"] >= risk_recency) and (row["Monetary"] >= risk_value_floor):
                return "At Risk"
            return "Standard"

        df_rfm["Segment"] = df_rfm.apply(segment, axis=1)

        risk_users = df_rfm[df_rfm["Segment"] == "At Risk"]
        vip_users = df_rfm[df_rfm["Segment"] == "VIP"]
        risk_value = risk_users["Monetary"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VIP 人數", f"{len(vip_users)} 人")
        m2.metric("流失預警人數", f"{len(risk_users)} 人", delta="需挽回", delta_color="inverse")
        m3.metric("潛在流失金額", f"€{risk_value:,.0f}")
        m4.metric("At Risk 平均 Frequency", f"{risk_users['Frequency'].mean():.1f}")

        fig = px.scatter(
            df_rfm,
            x="Recency", y="Monetary",
            size="Frequency",
            color="Segment",
            title="RFM 客戶價值分佈圖（點越大=購買越頻繁）",
            hover_data=["CustomerID", "Recency", "Frequency", "Monetary"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Actionable Insight")
        st.success(
            f"建議針對 **At Risk（{len(risk_users)} 人）** 啟動 Win-back campaign（限時券/回購禮）。\n\n"
            f"可先用小規模 A/B test 驗證：例如 10% 抽樣投放 → 觀察回購率、客單、毛利是否顯著提升。"
        )

# =========================
# Tab 2: Inventory (Forecast & Ordering)
# =========================
with tab2:
    st.header("📦 需求預測與動態補貨 (Forecast & Ordering)")

    @st.cache_data
    def load_forecast_data(start="2026-01-01", periods=30, seed=7):
        np.random.seed(seed)
        dates = pd.date_range(start=start, periods=periods)
        base = 100
        # demo seasonality: weekend higher
        forecast = [
            int(base * (1.35 if d.dayofweek >= 5 else 1.0) + np.random.randint(-12, 12))
            for d in dates
        ]
        return pd.DataFrame({"Date": dates, "Forecast": forecast})

    df_inv = load_forecast_data()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("供應鏈參數")
        buffer_pct = st.slider("🛡️ 訂貨偏差 (Buffer %)", -20, 50, 10,
                               help="正數=多訂以避免缺貨；負數=少訂以降低報廢風險。")
        unit_cost = st.number_input("進貨成本 (€/unit)", min_value=0.0, value=0.5, step=0.1)
        overstock_loss_rate = st.slider("🥬 過量訂貨損失率 (%)", 0, 100, 60,
                                        help="多訂不等於全報廢；可視為折價/報廢/耗損比例。")
        lost_margin = st.number_input("🚫 缺貨損失（毛利）(€/unit)", min_value=0.0, value=0.8, step=0.1)

        with st.expander("Methodology & assumptions"):
            st.markdown(
                "- 目前 Forecast 為**示範用季節性模擬**（週末需求較高）。\n"
                "- 你可把 Forecast 替換成 Prophet/ARIMA/ML 預測輸出。\n"
                "- 成本估算：\n"
                "  - 過量：overstock_units × unit_cost × loss_rate\n"
                "  - 缺貨：understock_units × lost_margin（用毛利近似缺貨損失）"
            )

    with col2:
        df_inv["Order_Qty"] = df_inv["Forecast"] * (1 + buffer_pct / 100)

        # costs
        overstock_units = np.maximum(df_inv["Order_Qty"] - df_inv["Forecast"], 0)
        understock_units = np.maximum(df_inv["Forecast"] - df_inv["Order_Qty"], 0)

        overstock_cost = (overstock_units * unit_cost * (overstock_loss_rate / 100)).sum()
        stockout_cost = (understock_units * lost_margin).sum()
        total_cost = overstock_cost + stockout_cost

        m1, m2, m3 = st.columns(3)
        m1.metric("建議總訂貨量", f"{int(df_inv['Order_Qty'].sum()):,}", delta=f"{buffer_pct:+d}% vs forecast")
        m2.metric("過量成本（折價/報廢）", f"€{overstock_cost:,.0f}", delta_color="inverse")
        m3.metric("缺貨損失（毛利）", f"€{stockout_cost:,.0f}", delta_color="inverse")

        st.caption(f"Total risk cost (Overstock + Stockout) ≈ €{total_cost:,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_inv["Date"], y=df_inv["Forecast"], name="需求預測（Forecast）"))
        fig.add_trace(go.Scatter(x=df_inv["Date"], y=df_inv["Order_Qty"], name="建議訂貨量（Order）",
                                 line=dict(dash="dash")))
        fig.update_layout(title="Forecast vs Ordering Plan")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Actionable Insight")
        st.info(
            "你可以把 Buffer 當成「服務水準 vs 報廢」的旋鈕：\n"
            "- 想降低缺貨：提高 Buffer（但過量成本上升）\n"
            "- 想降低報廢：降低 Buffer（但缺貨損失可能上升）\n\n"
            "面試時一句話：**我把補貨決策變成可調參的風險成本最小化問題。**"
        )

# =========================
# Tab 3: Basket (Association Rules)
# =========================
with tab3:
    st.header("🧺 購物籃交叉銷售策略 (Cross-Selling Strategy)")
    st.markdown("利用 **關聯規則 (Association Rules)** 找出「帶路雞」，以低毛利商品帶動高毛利營收。")

    col_ui, col_kpi = st.columns([1, 2])

    # demo rules DB (consistent units: €/unit margin)
    rules_db = {
        "Beer 🍺":    {"target": "Chips 🥔",   "support": 0.08, "confidence": 0.62, "lift": 5.0, "driver_margin": 0.10, "target_margin": 0.70, "desc": "週末狂歡組合"},
        "Milk 🥛":    {"target": "Bread 🍞",   "support": 0.12, "confidence": 0.41, "lift": 1.8, "driver_margin": 0.05, "target_margin": 0.35, "desc": "每日早餐剛需"},
        "Diapers 👶": {"target": "Beer 🍺",    "support": 0.03, "confidence": 0.28, "lift": 3.5, "driver_margin": 2.00, "target_margin": 0.10, "desc": "新手爸媽關聯購買"}
    }

    with col_ui:
        st.subheader("🔍 選擇帶路雞商品 (Driver)")
        selected_item = st.selectbox("請選擇促銷商品：", list(rules_db.keys()))
        rule = rules_db[selected_item]

        with st.expander("Methodology & assumptions"):
            st.markdown(
                "- 這裡的 association rules 為**示範用數值**。\n"
                "- Support/Confidence/Lift 在正式版應由交易資料計算。\n"
                "- Margin 統一用 **€/unit（每件毛利）**。"
            )

    with col_kpi:
        total_margin = rule["driver_margin"] + rule["target_margin"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("關聯商品 (Target)", rule["target"])
        k2.metric("Support", f"{rule['support']:.0%}")
        k3.metric("Confidence", f"{rule['confidence']:.0%}")
        k4.metric("Lift", f"{rule['lift']:.1f}x")

        st.metric("組合毛利（€/basket）", f"€{total_margin:.2f}")

        profit_data = pd.DataFrame({
            "Product": ["Driver (帶路雞)", "Target (被帶動)"],
            "Margin €/unit": [rule["driver_margin"], rule["target_margin"]],
        })
        fig_bar = px.bar(
            profit_data,
            x="Product", y="Margin €/unit",
            color="Product",
            title="單品毛利貢獻比較 (Margin Contribution)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 策略建議 (Actionable Insight)")

    if rule["driver_margin"] < rule["target_margin"]:
        st.success(
            f"**Loss Leader Strategy（帶路雞策略）**\n\n"
            f"- 洞察：{selected_item} 毛利較低，但能有效帶動 {rule['target']}（Lift {rule['lift']:.1f}x）。\n"
            f"- 行動：對 {selected_item} 做限時促銷／前段陳列，提升曝光與進店轉化。\n"
            f"- 目標：提高 **整體購物籃毛利**（用高毛利品 {rule['target']} 來補回）。"
        )
    else:
        st.info(
            f"**Bundle Strategy（組合策略）**\n\n"
            f"- 洞察：兩者皆具不錯毛利，且關聯度高。\n"
            f"- 行動：推出 bundle、同區陳列、或第二件折扣，提升客單價。"
        )

# =========================
# Tab 4: Pricing (Elasticity)
# =========================
with tab4:
    st.header("💰 價格彈性與獲利模擬 (Price Elasticity)")
    st.markdown("模擬 **價格變動** 對 **需求量** 的影響，尋找獲利最大化的甜蜜點。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("參數設定")
        base_price = st.number_input("目前售價 P0 (€)", 0.1, 500.0, 10.0, step=0.5)
        base_cost = st.number_input("商品成本 C (€)", 0.0, 499.0, 6.0, step=0.5)
        base_demand = st.number_input("目前日銷量 Q0", 1, 100000, 100, step=10)

        # Constant elasticity demand: Q = Q0 * (P/P0)^e, e < 0
        elasticity = st.slider(
            "價格彈性係數 e（負值）",
            -5.0, -0.1, -1.5, step=0.1,
            help="e 絕對值越大 → 價格越敏感；使用常彈性模型避免需求變成負數。"
        )

        with st.expander("Methodology & assumptions"):
            st.markdown(
                "- 需求模型：**Q = Q0 × (P/P0)^e**（常彈性模型，e < 0）。\n"
                "- 獲利：**(P - C) × Q**。\n"
                "- 若成本 ≥ 售價，獲利可能為負，屬正常提醒。"
            )

    with col2:
        price_change_pct = np.linspace(-0.2, 0.2, 60)
        sim_prices = base_price * (1 + price_change_pct)

        # constant elasticity demand
        sim_demand = base_demand * (sim_prices / base_price) ** (elasticity)

        sim_profit = (sim_prices - base_cost) * sim_demand

        max_idx = int(np.argmax(sim_profit))
        best_price = float(sim_prices[max_idx])
        best_profit = float(sim_profit[max_idx])

        st.metric("建議最佳售價", f"€{best_price:.2f}", delta=f"{(best_price-base_price)/base_price:+.1%}")
        st.metric("預估最大日獲利", f"€{best_profit:,.1f}")

        # Dual-axis plot for readability
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_prices, y=sim_profit, name="Profit (€)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=sim_prices, y=sim_demand, name="Demand (units)", mode="lines+markers", yaxis="y2"))

        fig.update_layout(
            title="Price vs Profit & Demand (Dual Axis)",
            xaxis_title="Price (€)",
            yaxis=dict(title="Profit (€)"),
            yaxis2=dict(title="Demand (units)", overlaying="y", side="right"),
        )
        fig.add_vline(x=best_price, line_dash="dash", annotation_text="Best Price")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Tab 5: Location (Geospatial)
# =========================
with tab5:
    st.header("🗺️ 客戶地理分佈 (Geospatial Insights)")
    st.markdown("分析目標地區的客戶密度，協助 **門市選址**、**自取點 (Pick-up Point)** 與 **物流配送** 決策。")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("地圖中心點")
        center_lat = st.number_input("Center Latitude", value=50.8503, format="%.6f")  # default Brussels
        center_lon = st.number_input("Center Longitude", value=4.3517, format="%.6f")
        n_points = st.slider("模擬客戶點數", 100, 5000, 500, step=100)

        with st.expander("Methodology & assumptions"):
            st.markdown(
                "- 地圖點位為 **Simulated customer pings**（示範用）。\n"
                "- 正式版可換成：會員地址、外送訂單座標、或區域彙總（zip/census）資料。"
            )

    with col1:
        @st.cache_data
        def load_geo_data(lat0, lon0, n, seed=11):
            np.random.seed(seed)
            lat = np.random.normal(lat0, 0.02, n)
            lon = np.random.normal(lon0, 0.02, n)
            return pd.DataFrame({"lat": lat, "lon": lon})

        df_map = load_geo_data(center_lat, center_lon, n_points)
        st.map(df_map)

    st.markdown("---")
    st.subheader("💡 商業洞察 (示範)")
    st.info(
        "你可以把地理頁面變成「選址決策」：\n"
        "- 熱區（密集客戶）→ 增設自取點 / 快送前置倉（dark store）\n"
        "- 稀疏區 → 以配送半徑/成本評估是否值得拓點\n\n"
        "正式版建議：用 hexbin/heatmap 顯示密度，並加入 2–3 個候選點 marker 做比較。"
    )
