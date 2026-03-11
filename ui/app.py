import sys
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Trading AI Dashboard", layout="wide", page_icon="📈")

def set_rtl():
    st.markdown(
        """
        <style>
        /* יישור כותרות וטקסט בלבד לימין */
        h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
            text-align: right;
            direction: rtl;
        }
        
        /* החרגה מפורשת של הטבלה - שתהיה משמאל לימין */
        [data-testid="stDataFrame"], .stDataFrame {
            direction: ltr !important;
            text-align: left !important;
        }

        /* יישור כרטיסיות האיתותים (Metrics) לימין */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
            text-align: right;
            direction: rtl;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
set_rtl()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.daily_predictor import get_daily_signals


st.title("מערכת חיזוי מניות אלגוריתמית 📈")
st.markdown("מערכת זו מבוססת על מודל Random Forest שלמד את התנהגות שוק ההון ומחשב הסתברויות קנייה לטווח של 5 ימים.")

if 'ticker_list' not in st.session_state:
    st.session_state.ticker_list = [
        "POLI.TA", "LUMI.TA", "ESLT.TA", "NICE.TA", "TEVA.TA", 
        "ICL.TA", "BEZQ.TA", "ENOG.TA", "ALHE.TA", "DSCT.TA",
        "NVMI.TA", "TSEM.TA", "AZRG.TA", "AMOT.TA", "PHOE.TA", 
        "FIBI.TA", "DANE.TA", "SPEN.TA", "MVNE.TA", "OPAL.TA"
    ]

st.subheader("ניהול יקום המניות (TA-35 + Custom)")
col1, col2 = st.columns([3, 1])

with col1:
    new_ticker = st.text_input("הכנס שם מניה להוספה (למשל ILDC.TA):").strip().upper()
with col2:
    st.write("") 
    st.write("")
    if st.button("➕ הוסף מניה", use_container_width=True):
        if new_ticker and new_ticker not in st.session_state.ticker_list:
            st.session_state.ticker_list.append(new_ticker)
            st.success(f"המניה {new_ticker} נוספה בהצלחה לרשימה!")
        elif new_ticker in st.session_state.ticker_list:
            st.warning("המניה כבר קיימת ברשימה.")

selected_tickers = st.multiselect(
    "המניות שינותחו כעת (לחץ על ה-X ליד מניה כדי להסיר אותה מהחישוב):",
    options=st.session_state.ticker_list,
    default=st.session_state.ticker_list
)

st.markdown("---")

if st.button("🚀 הפעל מודל חיזוי", use_container_width=True):
    if len(selected_tickers) < 5:
        st.error("שגיאה: נדרשות לפחות 5 מניות כדי לחשב את מדד מצב השוק (Market Return) בצורה אמינה עבור המודל.")
    else:
        with st.spinner("שואב נתונים היסטוריים עדכניים, מחשב מתנדים טכניים ומריץ את המודל..."):
            
            results_df = get_daily_signals(tickers=selected_tickers)
            
            if not results_df.empty:
                st.success("החישוב הושלם בהצלחה!")
                
                threshold = 0.55
                buy_signals = results_df[results_df['Buy_Probability'] >= threshold]
                
                if not buy_signals.empty:
                    st.subheader("🔥 איתותי קנייה חזקים (מעל 55% ביטחון)")
                    cols = st.columns(len(buy_signals))
                    for i, (idx, row) in enumerate(buy_signals.iterrows()):
                        with cols[i % len(cols)]:
                            st.metric(
                                label=row['Ticker'], 
                                value=f"{row['Buy_Probability']:.2%}", 
                                delta=f"RSI: {row['RSI_14']:.1f}"
                            )
                else:
                    st.info("אין איתותי קנייה חזקים להיום. Cash is a position too.")
                
                st.subheader("📊 דירוג המניות המלא")
                
                formatted_df = results_df.style.format({
                    'Close': "{:.2f}",
                    'Buy_Probability': "{:.2%}",
                    'RSI_14': "{:.2f}",
                    'Volume_Surge': "{:.2f}x",
                    'Daily_Volatility': "{:.2%}"
                }).background_gradient(subset=['Buy_Probability'], cmap='Greens')
                
                st.dataframe(formatted_df, use_container_width=True, height=400)
            else:
                st.error("אירעה שגיאה בשליפת הנתונים או בחישוב. אנא בדוק את הטרמינל לפרטים נוספים.")