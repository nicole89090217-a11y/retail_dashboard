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

# --- 1. 數據載入 (對齊零售常規) ---
@st.cache_data
def load_data_pro():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['itemDescription'] = df['itemDescription'].replace({'salty snack': 'Potato Chips'})
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 業界常規毛利設定
    margin_config = {
        'Bottled Beer': 0.08, 'Canned Beer': 0.07, 'Whole Milk': 0.05,
        'Potato Chips': 0.45, 'Soda': 0.40, 'Sausage': 0.35
    }
    df['Margin'] = df['Item'].map(margin_config).fillna(0.22)
    df['Price'] = df['Item'].map({'Bottled Beer': 18, 'Potato Chips': 5}).fillna(7.0)
    return df, margin_config

df_all, margin_lookup = load_data_pro()

# --- 2. 側邊欄 (全域變數) ---
st.sidebar.title("💎 營運策略控制台")
lead_time = st.sidebar.select_slider("物流前置時間 (Lead Time)", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_factor = st.sidebar.slider("安全庫存係數 (Buffer)", 1.0, 3.0, 1.65, help="調整服務水準")

# --- 3. 主要 UI 分頁 ---
st.title("🚀 Strategic Retail Executive Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["👥 RFM 行銷", "📦 需求預測與補貨", "🛍️ 策略判定 (MBA)", "📉 智慧定價"])

# --- Tab 2: 妳要的供應鏈模擬 (對齊截圖與描述) ---
with tab2:
    st.subheader("📦 供應鏈動態補貨與庫存平衡模擬")
    st.markdown(f"**當前模擬場景：** 考慮週末需求波動與季節性因素，前置時間設定為 **{lead_time} 天**。")
    
    # 建立時間序列模擬 (對齊截圖中的曲線感)
    time_series = df_all.groupby('Date')['Item'].count().tail(30).reset_index()
    time_series.columns = ['Date', 'Actual_Demand']
    
    # 模擬週末波動 (週末銷量 +30%)
    time_series['Is_Weekend'] = time_series['Date'].dt.dayofweek >= 5
    time_series['Projected_Demand'] = time_series['Actual_Demand'] * (1.3 if any(time_series['Is_Weekend']) else 1.0)
    
    # 計算補貨建議量 (加計 Buffer)
    avg_demand = time_series['Actual_Demand'].mean()
    std_demand = time_series['Actual_Demand'].std()
    buffer_amt = buffer_factor * std_demand * np.sqrt(lead_time)
    time_series['Reorder_Level'] = time_series['Projected_Demand'] + buffer_amt
    
    # 繪製動態曲線
    fig_supply = go.Figure()
    fig_supply.add_trace(go.Scatter(x=time_series['Date'], y=time_series['Actual_Demand'], name='實際歷史銷量', line=dict(color='#1F77B4', width=2)))
    fig_supply.add_trace(go.Scatter(x=time_series['Date'], y=time_series['Reorder_Level'], name='建議補貨水平 (加計 Buffer)', line=dict(color='#E63946', dash='dash')))
    fig_supply.update_layout(title="需求波動與動態訂貨量預測", width=None) # 自動填充
    st.plotly_chart(fig_supply, width='stretch')
    
    # 決策說明卡 (對齊妳的功能描述)
    c1, c2 = st.columns(2)
    with c1:
        st.warning("📉 **策略：提高服務水準 (High Buffer)**")
        st.write(f"- **優點：** 降低缺貨損失，保障週末營收。")
        st.write(f"- **風險：** 過量庫存成本上升，{('Sausage' if 'Sausage' in df_all['Item'].values else '生鮮品')} 報廢率增加。")
    with c2:
        st.success("📈 **策略：極致庫存周轉 (Low Buffer)**")
        st.write(f"- **優點：** 減少資金積壓與報廢損失。")
        st.write(f"- **風險：** 週末需求高峰可能導致缺貨損失，影響客單價（UPT）。")

    # 關鍵品項庫存狀態卡
    st.markdown("---")
    st.write("**核心監控品項庫存指標 (Beer & Chips Focus)：**")
    inv = df_all.groupby('Item').agg({'Member_number': 'count'})
    inv['Daily_Avg'] = inv['Member_number'] / df_all['Date'].nunique()
    inv['Safe_Buffer'] = (buffer_factor * (inv['Daily_Avg']*0.4) * np.sqrt(lead_time)).round(1)
    
    target_items = ['Bottled Beer', 'Potato Chips', 'Sausage']
    existing = [i for i in target_items if i in inv.index]
    if existing:
        m_cols = st.columns(len(existing))
        for i, item in enumerate(existing):
            m_cols[i].metric(item, f"{inv.loc[item, 'Daily_Avg']:.1f} /日", f"Buffer: {inv.loc[item, 'Safe_Buffer']}")

# --- Tab 3: 購物籃策略 (保持妳要的 Loss Leader / Bundle 判定) ---
with tab3:
    st.subheader("🛒 購物籃交叉銷售策略判定")
    basket = df_all.head(10000).groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number')
    basket_bool = (basket > 0).astype(bool)
    f_sets = apriori(basket_bool, min_support=0.005, use_colnames=True)
    rules = association_rules(f_sets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        sel_a = st.selectbox("選擇 Driver 商品 (A)：", sorted(rules['A'].unique()), index=0)
        top_rule = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        sel_b = top_rule['B']
        
        ma, mb = margin_lookup.get(sel_a, 0.20), margin_lookup.get(sel_b, 0.20)
        
        if ma <= 0.12 and mb >= 0.20:
            st.error("🚀 **建議策略：Loss Leader (引流策略)**")
            reason = f"商品 {sel_a} 毛利低，但能帶動高毛利商品 {sel_b} 的銷售。"
        elif top_rule['lift'] > 1.2:
            st.success("📦 **建議策略：Bundle (綑綁銷售)**")
            reason = f"兩者具備強關聯 (Lift: {top_rule['lift']:.2f})，適合組合銷售。"
        else:
            st.info("💡 **建議策略：Cross-Sell (交叉銷售)**")
            reason = "建議進行一般陳列推薦。"
        
        st.markdown(f"""<div style="background-color:rgba(100,100,100,0.1); padding:15px; border-radius:10px; border-left: 5px solid #ccc;"><b>商業診斷：</b>{reason}</div>""", unsafe_allow_html=True)
        st.plotly_chart(px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift'), width='stretch')

# (Tab 1 & 4 之前寫過的內容可直接補上)
