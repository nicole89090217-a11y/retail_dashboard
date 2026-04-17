import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 環境設置 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Center", page_icon="📊", layout="wide")

# --- 1. 核心數據引擎：強植毛利矩陣 ---
@st.cache_data
def load_and_refine_data_v7():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # --- 這裡就是妳要的毛利核心數據 ---
    # 我們設定啤酒為低毛利(引流)，零食與肉類為高毛利(獲利)
    margin_config = {
        'Bottled Beer': 0.08,   # 8% (超級帶路雞)
        'Canned Beer': 0.10,
        'Salty Snack': 0.45,    # 45% (高獲利)
        'Sausage': 0.35,        # 35%
        'Soda': 0.40,           # 40%
        'Whole Milk': 0.05      # 5% (民生品)
    }
    
    # 將數據對齊
    df['Margin'] = df['Item'].map(margin_config).fillna(0.22) # 沒設定的預設 22%
    df['Price'] = df['Item'].map({'Bottled Beer': 18, 'Salty Snack': 5, 'Whole Milk': 3.5}).fillna(8.0)
    df['Profit'] = df['Price'] * df['Margin']
    
    return df, margin_config

df_all, margin_lookup = load_and_refine_data_v7()

# --- 2. 側邊欄 ---
st.sidebar.title("💎 策略控制中心")
lead_time = st.sidebar.select_slider("物流前置時間", options=[1, 3, 5, 7, 10, 14], value=5)
buffer_factor = st.sidebar.slider("安全庫存係數 (Buffer)", 1.0, 3.0, 1.65)

# --- 3. 主要分頁 ---
st.title("🚀 Strategic Retail Executive Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["👥 RFM 行銷", "📦 補貨預測", "🛍️ 策略判定 (MBA)", "📉 定價優化"])

# --- Tab 3: 這裡就是顯示毛利判斷的地方 ---
with tab3:
    st.subheader("🛒 購物籃交叉銷售策略 (基於關聯與毛利結構)")
    
    # 執行 MBA 運算
    df_mba = df_all.head(10000)
    basket_bool = df_mba.groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number') > 0
    f_sets = apriori(basket_bool.astype(bool), min_support=0.005, use_colnames=True)
    rules = association_rules(f_sets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        sel_a = st.selectbox("🎯 選擇 Driver 商品 (A)：", sorted(rules['A'].unique()), index=0)
        top_rule = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        sel_b = top_rule['B']
        
        # --- 決策引擎：毛利判斷 ---
        m_a = margin_lookup.get(sel_a, 0.22)
        m_b = margin_lookup.get(sel_b, 0.22)
        
        # 顯示毛利對照 (這行讓妳真的看到毛利！)
        st.write("---")
        c_m1, c_m2 = st.columns(2)
        c_m1.metric(f"{sel_a} 毛利率", f"{m_a*100:.0f}%", delta="- 低毛利" if m_a < 0.15 else None)
        c_m2.metric(f"{sel_b} 毛利率", f"{m_b*100:.0f}%", delta="+ 高獲利" if m_b > 0.25 else None)
        
        # 判定策略
        if m_a <= 0.12 and m_b >= 0.25:
            strategy, style = "Loss Leader (引流策略)", "error"
            reason = f"商品 {sel_a} 目前僅有 {m_a*100:.0f}% 毛利，適合作為引流品項，藉由高關聯性提升高毛利商品 {sel_b} ({m_b*100:.0f}%) 的銷量。"
        elif top_rule['lift'] > 2.0:
            strategy, style = "Bundle (綑綁銷售)", "success"
            reason = f"兩者毛利結構穩定，且具備極強關聯強度 (Lift: {top_rule['lift']:.2f})，適合進行綑綁行銷。"
        else:
            strategy, style = "Cross-Sell (交叉銷售)", "info"
            reason = "建議在陳列上進行鄰近擺放，以增加客戶轉化率。"

        # 輸出專業診斷卡
        if style == "error": st.error(f"🚀 **【建議策略：{strategy}】**\n\n{reason}")
        elif style == "success": st.success(f"💎 **【建議策略：{strategy}】**\n\n{reason}")
        else: st.info(f"💡 **【建議策略：{strategy}】**\n\n{reason}")

        st.plotly_chart(px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', orientation='h', color='lift'), width='stretch')
    else:
        st.warning("數據計算中...")

# (其餘 Tab 1, 2, 4 保持正常邏輯，只需確認 st.plotly_chart 使用 width='stretch')
