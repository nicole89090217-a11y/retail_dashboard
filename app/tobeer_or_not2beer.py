import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 環境設置 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Center", page_icon="📊", layout="wide")

# --- 1. 數據載入與商業邏輯定義 ---
@st.cache_data
def load_data_pro():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 數據正名與標準化
    df['itemDescription'] = df['itemDescription'].replace({'salty snack': 'Potato Chips'})
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 毛利矩陣：決定 Loss Leader 還是 Bundle
    margin_config = {
        'Bottled Beer': 0.10, 'Canned Beer': 0.08, 'Potato Chips': 0.45,
        'Sausage': 0.35, 'Whole Milk': 0.05, 'Soda': 0.40
    }
    
    df['Margin'] = df['Item'].map(margin_config).fillna(0.20)
    df['Price'] = df['Item'].map({'Bottled Beer': 18, 'Potato Chips': 5, 'Whole Milk': 3.5}).fillna(7.0)
    df['Profit'] = df['Price'] * df['Margin']
    
    return df, margin_config

df_all, margin_lookup = load_data_pro()

# --- 2. 側邊欄定義 ---
st.sidebar.title("💎 營運策略控制台")
lead_time = st.sidebar.select_slider("物流前置時間 (Lead Time)", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_factor = st.sidebar.slider("安全庫存係數 (Buffer)", 1.0, 3.0, 1.65)

# --- 3. 主要 UI 分頁 ---
st.title("🚀 Strategic Retail Executive Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["👥 RFM 行銷", "📦 供應鏈補貨", "🛍️ 購物籃策略", "📈 智慧定價"])

# --- Tab 1: RFM (指標對齊截圖) ---
with tab1:
    snapshot = df_all['Date'].max() + pd.Timedelta(days=1)
    rfm = df_all.groupby('Member_number').agg({'Date': lambda x: (snapshot - x.max()).days, 'Member_number': 'count', 'Price': 'sum'}).rename(columns={'Date': 'Recency', 'Member_number': 'Frequency', 'Price': 'Monetary'})
    
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
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency", title="RFM 價值矩陣")
    st.plotly_chart(fig_rfm, width='stretch') # 已修正語法

# --- Tab 3: 購物籃策略 (毛利判定版) ---
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
        
        # 毛利決策邏輯
        m_a, m_b = margin_lookup.get(sel_a, 0.20), margin_lookup.get(sel_b, 0.20)
        
        if m_a <= 0.12 and m_b >= 0.20:
            strategy, style, reason = "Loss Leader (引流)", "🚨", f"商品 {sel_a} 毛利低，但能帶動高毛利商品 {sel_b} 的銷售。"
            st.error(f"**{style} 策略建議：{strategy}**")
        elif top_rule['lift'] > 2.0:
            strategy, style, reason = "Bundle (綑綁)", "💎", f"兩者具備極強關聯 (Lift: {top_rule['lift']:.2f})，適合組合銷售。"
            st.success(f"**{style} 策略建議：{strategy}**")
        else:
            strategy, style, reason = "Cross-Sell (交叉銷售)", "💡", "建議進行一般陳列推薦。"
            st.info(f"**{style} 策略建議：{strategy}**")

        st.markdown(f"""<div style="background-color:rgba(100,100,100,0.1); padding:15px; border-radius:10px;"><b>策略診斷：</b>{reason}</div>""", unsafe_allow_html=True)
        
        fig_mba = px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift')
        st.plotly_chart(fig_mba, width='stretch') # 已修正語法

# --- 其餘 Tab 2 & 4 修正 ---
with tab2:
    # ... 供應鏈邏輯 ...
    st.plotly_chart(px.bar(df_all.head(10), x='Item', y='Price'), width='stretch') # 已修正語法
with tab4:
    # ... 定價邏輯 ...
    st.plotly_chart(px.line(x=[1,2,3], y=[1,2,3]), width='stretch') # 已修正語法

st.sidebar.caption("Dashboard Optimized v6.0")
