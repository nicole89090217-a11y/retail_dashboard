import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 環境淨化 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Dash", page_icon="🍺", layout="wide")

# --- 1. 數據載入與商業維度擴增 ---
@st.cache_data
def load_and_refine_data():
    # 使用真實超市交易大數據
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 【關鍵修正】強制將數據中的零食更名為洋芋片
    df['itemDescription'] = df['itemDescription'].replace({
        'salty snack': 'Potato Chips',
        'bottled beer': 'Bottled Beer',
        'canned beer': 'Canned Beer'
    })
    
    # 模擬利潤與成本矩陣 (用於策略判定)
    # 設定：啤酒與洋芋片毛利不同，用來展示 Loss Leader vs Bundle
    margin_map = {
        'Bottled Beer': 0.15,  # 低毛利 (引流)
        'Canned Beer': 0.12,
        'Potato Chips': 0.45,  # 高毛利 (獲利)
        'Whole Milk': 0.05,
        'Sausage': 0.35
    }
    
    # 賦予價格與利潤
    np.random.seed(42)
    df['Price'] = df['itemDescription'].map({'Bottled Beer': 18, 'Potato Chips': 5, 'Whole Milk': 3}).fillna(np.random.uniform(4, 10))
    df['Margin'] = df['itemDescription'].map(margin_map).fillna(0.25)
    df['Profit'] = df['Price'] * df['Margin']
    df['Cost'] = df['Price'] - df['Profit']
    
    return df

df_all = load_and_clean_data() if 'load_and_clean_data' in locals() else load_and_refine_data()

# --- 2. 主要 UI 與分頁 ---
st.title("📊 Strategic Retail Operation Center")
tab1, tab2, tab3, tab4 = st.tabs(["👥 客戶精準行銷 (RFM)", "🚚 需求預測 (Supply Chain)", "🛍️ 交叉銷售 (MBA)", "📉 智慧定價 (Elasticity)"])

# --- 模組 1: 客戶精準行銷 (RFM) ---
with tab1:
    st.subheader("🎯 客戶分群與流失預警")
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Member_number': 'count',
        'Price': 'sum'
    }).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
    # 定義分群邏輯
    def segment_customer(df):
        if df['Monetary'] > rfm['Monetary'].quantile(0.8) and df['Frequency'] > rfm['Frequency'].quantile(0.8):
            return 'VIP'
        elif df['Recency'] > 30 and (df['Monetary'] > rfm['Monetary'].median()):
            return 'At Risk'
        else:
            return 'Standard'
            
    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    
    # 顯示妳要求的關鍵指標
    at_risk_df = rfm[rfm['Segment'] == 'At Risk']
    col1, col2, col3 = st.columns(3)
    col1.metric("流失預警客戶數", f"{len(at_risk_df)} 人")
    col2.metric("潛在流失總金額", f"€{at_risk_df['Monetary'].sum():,.0f}")
    col3.metric("流失前平均消費次數", f"{at_risk_df['Frequency'].mean():.1f} 次")
    
    st.plotly_chart(px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency", title="RFM 客戶價值矩陣"), use_container_width=True)
    st.info(f"💡 **行銷建議：** 針對 {len(at_risk_df)} 名 At Risk 客戶啟動限時券，這群人過去貢獻高但已超過 30 天未現身。")

# --- 模組 2: 補貨與物流 (Supply Chain) ---
with tab2:
    st.subheader("📦 供應鏈補貨 Buffer 管理")
    buffer_val = st.slider("調整安全庫存 Buffer (旋鈕)", 0.5, 3.0, 1.65, help="提高 Buffer 降低缺貨風險，但增加持有成本")
    
    inv = df_all.groupby('itemDescription').agg({'Member_number': 'count'})
    inv['Daily_Avg'] = inv['Member_number'] / df_all['Date'].nunique()
    inv['Std_Dev'] = inv['Daily_Avg'] * 0.5 # 模擬需求波動
    
    # 計算安全庫存
    inv['Safety_Stock'] = (buffer_val * inv['Std_Dev'] * np.sqrt(7)).round(1) # 以 7 天 Lead time 為例
    
    targets = ['Bottled Beer', 'Potato Chips', 'Sausage', 'Whole Milk']

# 找出數據中實際存在的品項（避免 KeyError）
existing_targets = [item for item in targets if item in inv.index]

if existing_targets:
    target_inv = inv.loc[existing_targets].reset_index()
    
    # 這裡放原本的 fig_inv 繪圖代碼
    fig_inv = px.bar(target_inv, x='Item', y=['Daily_Avg', 'Safety_Stock'],
                    title=f"前置時間 {lead_time} 天下的安全庫存組成",
                    barmode='group', color_discrete_sequence=['#457b9d', '#e63946'])
    st.plotly_chart(fig_inv, use_container_width=True)
else:
    st.warning("⚠️ 在數據中找不到指定的品項（Bottled Beer, Potato Chips 等），請檢查數據載入狀況。")
    fig_inv = px.bar(target_inv, x='itemDescription', y=['Daily_Avg', 'Safety_Stock'], 
                     title="需求量 vs. 補貨 Buffer", barmode='group', labels={'value': '數量'})
    st.plotly_chart(fig_inv, use_container_width=True)
    
    c1, c2 = st.columns(2)
    c1.warning("⚠️ 提高 Buffer：缺貨損失 ↓ | 過量成本 ↑")
    c2.success("✅ 降低 Buffer：報廢損耗 ↓ | 缺貨風險 ↑")

# --- 模組 3: 購物籃交叉銷售 (MBA) ---
with tab3:
    st.subheader("🛒 購物籃交叉銷售策略")
    
    # 建立矩陣
    df_mba = df_all.head(15000) # 取樣本確保運算速度
    basket = df_mba.groupby(['Member_number', 'itemDescription'])['itemDescription'].count().unstack().reset_index().fillna(0).set_index('Member_number')
    basket_sets = (basket > 0).astype(bool)
    
    frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        sel_a = st.selectbox("選擇 Driver 商品：", sorted(rules['A'].unique()), index=sorted(rules['A'].unique()).index('Bottled Beer'))
        best_rule = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        
        # 策略判斷邏輯
        # 假設如果 Driver 毛利 < 0.2 則判定為 Loss Leader
        is_low_margin = (sel_a == 'Bottled Beer') # 這裡手動標記展示
        strategy = "Loss Leader (引流)" if is_low_margin else "Bundle (綑綁)"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("建議策略", strategy)
        c2.metric("搭配商品", best_rule['B'])
        c3.metric("提升率 (Lift)", f"{best_rule['lift']:.2f}x")
        
        st.write(f"**策略解釋：** 買了 {sel_a} 的人買 {best_rule['B']} 的機率高出許多。")
        st.dataframe(rules[['A', 'B', 'support', 'confidence', 'lift']].head(10))

# --- 模組 4: 智慧定價 (Price Elasticity) ---
with tab4:
    st.subheader("💰 價格彈性與利潤優化")
    col_p1, col_p2 = st.columns(2)
    
    cost = col_p1.number_input("輸入採購成本 (Cost)", value=10.0)
    current_price = col_p1.number_input("當前售價 (RSP)", value=15.0)
    elasticity = col_p2.slider("彈性係數 (Elasticity)", 0.0, 5.0, 2.4)
    
    # 模擬利潤曲線
    price_range = np.linspace(cost * 1.1, current_price * 2, 50)
    # 銷量模擬: Q = Q0 * (P/P0)^-E
    demand = 100 * (price_range / current_price) ** -elasticity
    profit_curve = (price_range - cost) * demand
    
    opt_price = price_range[np.argmax(profit_curve)]
    
    fig_p = px.line(x=price_range, y=profit_curve, title="利潤最大化模擬曲線", labels={'x': '售價', 'y': '預期總利潤'})
    fig_p.add_vline(x=opt_price, line_dash="dash", line_color="green", annotation_text=f"最佳售價: €{opt_price:.1f}")
    st.plotly_chart(fig_p, use_container_width=True)
    st.success(f"🎯 **優化結果：** 根據彈性係數 {elasticity}，建議售價應調整為 **€{opt_price:.1f}** 以獲取最大利潤。")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard engineered by Yi-Han | Data Strategy Specialist")
