import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 0. 環境設置 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Strategic Retail Center", page_icon="📊", layout="wide")

# --- 1. 數據載入與「毛利結構」定義 ---
@st.cache_data
def load_data_with_margins():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Item'] = df['itemDescription'].str.strip().str.title()
    
    # 【關鍵：毛利矩陣】定義不同品類的利潤空間
    # 啤酒通常作為引流 (低毛利)，肉類/零食作為獲利 (高毛利)
    margin_config = {
        'Bottled Beer': 0.10,   # 10% (低毛利，帶路雞預選)
        'Canned Beer': 0.08,
        'Sausage': 0.35,        # 35% (獲利款)
        'Whole Milk': 0.05,     # 5% (民生必需品)
        'Yogurt': 0.25,
        'Soda': 0.40,           # 40% (高毛利)
        'Pastry': 0.30
    }
    
    df['Margin'] = df['Item'].map(margin_config).fillna(0.20)
    df['Price'] = df['Item'].map({'Bottled Beer': 18, 'Sausage': 12, 'Whole Milk': 3.5}).fillna(7.0)
    df['Profit'] = df['Price'] * df['Margin']
    
    return df, margin_config

df_all, margin_lookup = load_data_with_margins()

# (側邊欄與前兩個 Tab 保持不變，重點看 Tab 3)

# --- Tab 3: 購物籃分析 (新增毛利策略判定) ---
with st.container(): # 這裡假設是在 Tab 3 的內容中
    st.subheader("🛒 購物籃交叉銷售策略 (基於關聯性與毛利)")
    
    # 建立 MBA 矩陣
    basket_bool = df_all.head(10000).groupby(['Member_number', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Member_number') > 0
    frequent_itemsets = apriori(basket_bool.astype(bool), min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['A'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['B'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        sel_a = st.selectbox("選擇 Driver 商品 (A)：", sorted(rules['A'].unique()), index=0)
        
        # 抓取該規則的數據
        top_rule = rules[rules['A'] == sel_a].sort_values('lift', ascending=False).iloc[0]
        sel_b = top_rule['B']
        
        # --- 核心判定邏輯：毛利判定 ---
        margin_a = margin_lookup.get(sel_a, 0.20)
        margin_b = margin_lookup.get(sel_b, 0.20)
        
        # 1. 如果 Driver (A) 毛利低於 12% 且搭配商品 (B) 毛利高於 20%
        if margin_a <= 0.12 and margin_b >= 0.20:
            strategy = "Loss Leader (引流策略)"
            color = "inverse"
            reason = f"商品 {sel_a} 毛利低 ({margin_a*100:.0f}%)，但能顯著帶動高毛利商品 {sel_b} ({margin_b*100:.0f}%) 的銷售。"
        # 2. 如果兩者毛利皆不低，且 Lift 強
        elif top_rule['lift'] > 2.0:
            strategy = "Bundle (綑綁銷售)"
            color = "normal"
            reason = f"兩者毛利穩定且具備極強關聯 (Lift: {top_rule['lift']:.2f})，適合打包銷售提升客單價。"
        else:
            strategy = "Cross-Sell (交叉銷售)"
            color = "off"
            reason = "建議在結帳頁面或貨架鄰近處進行一般陳列推薦。"

        # 顯示結果
        m1, m2, m3 = st.columns(3)
        m1.metric("建議策略", strategy)
        m2.metric("搭配商品 (B)", sel_b)
        m3.metric("提升率 (Lift)", f"{top_rule['lift']:.2f}x")
        
        st.help(f"**策略診斷：** {reason}")
        
        # 繪圖
        fig_mba = px.bar(rules[rules['A'] == sel_a].head(5), x='lift', y='B', color='lift',
                         orientation='h', title=f"與 {sel_a} 關聯最強的 Top 5 品項")
        st.plotly_chart(fig_mba, use_container_width=True)

# (其餘 Tab 4 價格彈性代碼...)
