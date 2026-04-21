"""
BitScope | Bloomberg Terminal Dashboard
========================================
Pitch-black terminal aesthetic with gold/green/red accents.
AI Analyst layer powered by Anthropic Claude API.
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BitScope Terminal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# TERMINAL CSS — Bloomberg / Binance aesthetic
# ─────────────────────────────────────────────
TERMINAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-void:    #000000;
    --bg-panel:   #080c10;
    --bg-card:    #0d1117;
    --bg-border:  #1a2433;
    --gold:       #f0b429;
    --gold-dim:   #a07820;
    --green:      #00e676;
    --green-dim:  #00994d;
    --red:        #ff1744;
    --red-dim:    #99001a;
    --blue:       #29b6f6;
    --blue-dim:   #1a6e99;
    --text-prime: #e8eaf0;
    --text-muted: #546e7a;
    --text-dim:   #37474f;
    --font-mono:  'Share Tech Mono', monospace;
    --font-head:  'Orbitron', monospace;
    --font-body:  'Rajdhani', sans-serif;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: var(--bg-void) !important; color: var(--text-prime); }

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-panel) !important; border-right: 1px solid var(--bg-border); }
section[data-testid="stMain"] > div { padding-top: 0 !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: var(--font-head) !important; }
p, div, span, label { font-family: var(--font-body) !important; }
code, pre { font-family: var(--font-mono) !important; }

/* ── Scanline overlay ── */
body::after {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
    );
}

/* ── Top navigation bar ── */
.terminal-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 32px; background: var(--bg-panel);
    border-bottom: 1px solid var(--gold-dim);
    position: sticky; top: 0; z-index: 100;
}
.terminal-nav .brand {
    font-family: var(--font-head); font-size: 1.4rem; font-weight: 900;
    color: var(--gold); letter-spacing: 4px; text-shadow: 0 0 20px var(--gold-dim);
}
.terminal-nav .brand span { color: var(--text-muted); font-size: 0.7rem; font-weight: 400; display: block; letter-spacing: 6px; }
.terminal-nav .timestamp { font-family: var(--font-mono); color: var(--text-muted); font-size: 0.8rem; }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--bg-border); margin: 1px; }
.metric-card {
    background: var(--bg-card); padding: 20px 24px;
    border-left: 3px solid transparent; transition: border-color 0.2s;
}
.metric-card:hover { border-left-color: var(--gold); }
.metric-card .label { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
.metric-card .value { font-family: var(--font-head); font-size: 1.6rem; font-weight: 700; color: var(--text-prime); }
.metric-card .delta { font-family: var(--font-mono); font-size: 0.8rem; margin-top: 4px; }
.delta-up   { color: var(--green); }
.delta-down { color: var(--red); }
.delta-neu  { color: var(--text-muted); }

/* ── Section headers ── */
.section-header {
    font-family: var(--font-mono); font-size: 0.7rem; color: var(--gold-dim);
    text-transform: uppercase; letter-spacing: 4px;
    border-bottom: 1px solid var(--bg-border); padding-bottom: 8px; margin: 24px 0 16px;
    display: flex; align-items: center; gap: 10px;
}
.section-header::before { content: '//'; color: var(--text-dim); }

/* ── AI Analyst Panel ── */
.analyst-panel {
    background: linear-gradient(135deg, #080c10 0%, #0a1020 100%);
    border: 1px solid var(--bg-border); border-left: 4px solid var(--gold);
    padding: 24px 28px; margin: 8px 0;
    position: relative; overflow: hidden;
}
.analyst-panel::before {
    content: 'AI ANALYST';
    position: absolute; top: 12px; right: 16px;
    font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-dim);
    letter-spacing: 3px;
}
.analyst-panel .verdict {
    font-family: var(--font-head); font-size: 1.8rem; font-weight: 900;
    letter-spacing: 3px; margin-bottom: 12px;
}
.verdict-bull { color: var(--green); text-shadow: 0 0 30px rgba(0,230,118,0.4); }
.verdict-bear { color: var(--red); text-shadow: 0 0 30px rgba(255,23,68,0.4); }
.analyst-panel .reasoning { font-family: var(--font-body); font-size: 1.05rem; color: var(--text-prime); line-height: 1.7; }
.analyst-panel .confidence {
    font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);
    margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--bg-border);
}

/* ── Signal badges ── */
.signal-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0; }
.badge {
    font-family: var(--font-mono); font-size: 0.7rem; padding: 5px 12px;
    border: 1px solid; letter-spacing: 2px;
}
.badge-bull  { color: var(--green); border-color: var(--green-dim); background: rgba(0,230,118,0.05); }
.badge-bear  { color: var(--red);   border-color: var(--red-dim);   background: rgba(255,23,68,0.05); }
.badge-neu   { color: var(--blue);  border-color: var(--blue-dim);  background: rgba(41,182,246,0.05); }
.badge-gold  { color: var(--gold);  border-color: var(--gold-dim);  background: rgba(240,180,41,0.05); }

/* ── Footer ── */
.terminal-footer {
    text-align: center; padding: 24px;
    font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-dim);
    border-top: 1px solid var(--bg-border); margin-top: 40px; letter-spacing: 2px;
}

/* Streamlit element overrides */
.stSpinner > div { border-top-color: var(--gold) !important; }
[data-testid="stMetric"] { display: none; }   /* we use custom cards */
div[data-testid="column"] { padding: 0 4px; }
</style>
"""

# ─────────────────────────────────────────────
# DATA & MODEL LOADER
# ─────────────────────────────────────────────

class BitScopeApp:

    FEATURES = [
        "Close", "Volume", "sentiment", "news_volume",
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Width", "BB_Pct", "ATR", "OBV",
        "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
        "return_lag1", "return_lag2", "return_lag3",
        "volatility_7d",
    ]

    @staticmethod
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/processed/final_dataset.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    @staticmethod
    @st.cache_resource
    def load_model():
        model = joblib.load("models/bitscope_xgb.pkl")
        scaler = joblib.load("models/bitscope_scaler.pkl")
        return model, scaler

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_headlines():
        try:
            with open("data/raw/news.json") as f:
                news = json.load(f)
            return [n["title"] for n in news[:5]]
        except Exception:
            return ["No recent headlines found."]


# ─────────────────────────────────────────────
# AI ANALYST ENGINE
# ─────────────────────────────────────────────

def generate_ai_reasoning(prediction: int, headlines: list[str], rsi: float, macd_hist: float, sentiment: float) -> str:
    """Calls Claude claude-sonnet-4-20250514 to generate a 2-sentence analyst note."""

    direction = "BULLISH (price expected to rise)" if prediction == 1 else "BEARISH (price expected to fall)"
    headline_block = "\n".join(f"- {h}" for h in headlines[:5])

    prompt = f"""You are a senior quantitative analyst at a crypto hedge fund writing a brief market note.

Model prediction: {direction}
RSI: {rsi:.1f}
MACD Histogram: {macd_hist:.4f}
Sentiment score: {sentiment:.3f}

Top headlines:
{headline_block}

Write exactly 2 concise, sophisticated sentences explaining the market reasoning behind this prediction. 
Use technical language appropriate for a Bloomberg Terminal. Do not use bullet points. 
Do not restate the prediction — explain the *why* using the data above."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 180,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        data = resp.json()
        return data["content"][0]["text"].strip()
    except Exception as e:
        return f"AI Analyst unavailable: {e}"


# ─────────────────────────────────────────────
# PLOTLY TERMINAL CHART
# ─────────────────────────────────────────────

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#080c10",
    font=dict(family="Share Tech Mono", color="#546e7a", size=11),
    margin=dict(l=0, r=0, t=32, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2433", borderwidth=1,
                font=dict(size=10), orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    xaxis=dict(gridcolor="#0d1520", linecolor="#1a2433", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0d1520", linecolor="#1a2433", showgrid=True, zeroline=False),
)

def build_main_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        vertical_spacing=0.02,
        subplot_titles=("BTC / USD", "RSI (14)", "MACD", "Sentiment"),
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00e676", decreasing_fillcolor="#ff1744",
        name="BTC/USD", line=dict(width=1),
    ), row=1, col=1)

    # Bollinger Bands overlay
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], line=dict(color="#1a2433", width=1), name="BB Upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], line=dict(color="#1a2433", width=1), fill="tonexty", fillcolor="rgba(41,182,246,0.04)", name="Bollinger Bands"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Mid"], line=dict(color="#29b6f6", width=1, dash="dot"), name="BB Mid", showlegend=False), row=1, col=1)

    # ── RSI ──
    rsi_color = df["RSI"].apply(lambda v: "#ff1744" if v > 70 else ("#00e676" if v < 30 else "#f0b429"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#f0b429", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff1744", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00e676", line_width=1, row=2, col=1)

    # ── MACD ──
    hist_colors = ["#00e676" if v >= 0 else "#ff1744" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], marker_color=hist_colors, name="MACD Hist", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], line=dict(color="#29b6f6", width=1.5), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], line=dict(color="#f0b429", width=1, dash="dot"), name="Signal"), row=3, col=1)

    # ── Sentiment ──
    sent_colors = ["#00e676" if v > 0.05 else ("#ff1744" if v < -0.05 else "#546e7a") for v in df["sentiment"]]
    fig.add_trace(go.Bar(x=df["Date"], y=df["sentiment"], marker_color=sent_colors, name="Sentiment", opacity=0.8), row=4, col=1)

    fig.update_layout(**CHART_LAYOUT, height=720)
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Subplot-specific styling
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#0d1520", linecolor="#1a2433", row=i, col=1)
        fig.update_yaxes(gridcolor="#0d1520", linecolor="#1a2433", row=i, col=1)

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.family = "Share Tech Mono"
        ann.font.size = 10
        ann.font.color = "#546e7a"

    return fig


def build_obv_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["OBV"],
        fill="tozeroy", line=dict(color="#f0b429", width=1.5),
        fillcolor="rgba(240,180,41,0.06)", name="OBV",
    ))
    fig.update_layout(**CHART_LAYOUT, height=200, title=dict(text="ON-BALANCE VOLUME", font=dict(size=10, color="#546e7a")))
    return fig


# ─────────────────────────────────────────────
# SIGNAL BADGES HELPER
# ─────────────────────────────────────────────

def signal_badges(rsi, macd_hist, sentiment, bb_pct):
    badges = []
    # RSI
    if rsi > 70:   badges.append(('<span class="badge badge-bear">RSI OVERBOUGHT</span>', "bear"))
    elif rsi < 30: badges.append(('<span class="badge badge-bull">RSI OVERSOLD</span>', "bull"))
    else:          badges.append(('<span class="badge badge-neu">RSI NEUTRAL</span>', "neu"))
    # MACD
    if macd_hist > 0: badges.append(('<span class="badge badge-bull">MACD BULLISH CROSS</span>', "bull"))
    else:             badges.append(('<span class="badge badge-bear">MACD BEARISH CROSS</span>', "bear"))
    # Sentiment
    if sentiment > 0.1:    badges.append(('<span class="badge badge-bull">SENTIMENT POSITIVE</span>', "bull"))
    elif sentiment < -0.1: badges.append(('<span class="badge badge-bear">SENTIMENT NEGATIVE</span>', "bear"))
    else:                  badges.append(('<span class="badge badge-neu">SENTIMENT NEUTRAL</span>', "neu"))
    # BB position
    if bb_pct > 0.8:   badges.append(('<span class="badge badge-gold">BB UPPER BAND PRESSURE</span>', "gold"))
    elif bb_pct < 0.2: badges.append(('<span class="badge badge-bull">BB LOWER BAND SUPPORT</span>', "bull"))

    return "".join(b[0] for b in badges)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

    # ── Top Nav ──
    st.markdown(f"""
    <div class="terminal-nav">
        <div class="brand">BITSCOPE<span>MARKET INTELLIGENCE TERMINAL</span></div>
        <div class="timestamp">⬤ LIVE &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d  %H:%M:%S UTC')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ──
    try:
        df = BitScopeApp.load_data()
        model, scaler = BitScopeApp.load_model()
    except Exception as e:
        st.error(f"Failed to load data or model: {e}")
        st.info("Run `process_data.py` then `train_model.py` first.")
        st.stop()

    headlines = BitScopeApp.load_headlines()
    latest = df.iloc[-1]

    # ── Key Metrics ──
    price_delta = latest["Close"] - df["Close"].iloc[-2]
    price_pct   = price_delta / df["Close"].iloc[-2] * 100
    delta_cls   = "delta-up" if price_delta >= 0 else "delta-down"
    delta_arrow = "▲" if price_delta >= 0 else "▼"
    rsi_val     = latest.get("RSI", float("nan"))
    sent_val    = latest.get("sentiment", 0)
    atr_val     = latest.get("ATR", float("nan"))

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">BTC / USD</div>
            <div class="value">${latest['Close']:,.2f}</div>
            <div class="delta {delta_cls}">{delta_arrow} ${abs(price_delta):,.2f} &nbsp;({price_pct:+.2f}%)</div>
        </div>
        <div class="metric-card">
            <div class="label">RSI (14)</div>
            <div class="value">{rsi_val:.1f}</div>
            <div class="delta {'delta-down' if rsi_val > 70 else ('delta-up' if rsi_val < 30 else 'delta-neu')}">
                {'OVERBOUGHT' if rsi_val > 70 else ('OVERSOLD' if rsi_val < 30 else 'NEUTRAL')}
            </div>
        </div>
        <div class="metric-card">
            <div class="label">Sentiment Score</div>
            <div class="value">{sent_val:+.3f}</div>
            <div class="delta {'delta-up' if sent_val > 0.05 else ('delta-down' if sent_val < -0.05 else 'delta-neu')}">
                {'BULLISH' if sent_val > 0.05 else ('BEARISH' if sent_val < -0.05 else 'NEUTRAL')}
            </div>
        </div>
        <div class="metric-card">
            <div class="label">ATR (Volatility)</div>
            <div class="value">${atr_val:,.0f}</div>
            <div class="delta delta-neu">DAILY RANGE ESTIMATE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Analyst ──
    st.markdown('<div class="section-header">AI ANALYST &nbsp; PRICE DIRECTION PREDICTION</div>', unsafe_allow_html=True)

    available_features = [f for f in BitScopeApp.FEATURES if f in df.columns]
    latest_features = df[available_features].iloc[[-1]]
    latest_features_sc = scaler.transform(latest_features)
    prediction = model.predict(latest_features_sc)[0]
    proba = model.predict_proba(latest_features_sc)[0]
    confidence = max(proba) * 100

    verdict_cls  = "verdict-bull" if prediction == 1 else "verdict-bear"
    verdict_text = "▲ LONG SIGNAL — UPWARD TREND" if prediction == 1 else "▼ SHORT SIGNAL — DOWNWARD TREND"

    with st.spinner("Generating AI market analysis..."):
        reasoning = generate_ai_reasoning(
            prediction, headlines,
            rsi=rsi_val, macd_hist=latest.get("MACD_Hist", 0),
            sentiment=sent_val,
        )

    badges_html = signal_badges(rsi_val, latest.get("MACD_Hist", 0), sent_val, latest.get("BB_Pct", 0.5))

    st.markdown(f"""
    <div class="analyst-panel">
        <div class="verdict {verdict_cls}">{verdict_text}</div>
        <div class="signal-row">{badges_html}</div>
        <div class="reasoning">{reasoning}</div>
        <div class="confidence">
            MODEL CONFIDENCE: {confidence:.1f}% &nbsp;|&nbsp;
            FEATURES: {len(available_features)} &nbsp;|&nbsp;
            ENGINE: XGBoost + GridSearch CV &nbsp;|&nbsp;
            UPDATED: {datetime.now().strftime('%H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ──
    st.markdown('<div class="section-header">PRICE CHART &nbsp; CANDLESTICK + INDICATORS</div>', unsafe_allow_html=True)

    chart_window = st.select_slider("Lookback Window", options=["30D", "60D", "90D", "180D", "ALL"], value="90D")
    window_map = {"30D": 30, "60D": 60, "90D": 90, "180D": 180, "ALL": len(df)}
    chart_df = df.tail(window_map[chart_window])

    st.plotly_chart(build_main_chart(chart_df), use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(build_obv_chart(chart_df), use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">TOP HEADLINES</div>', unsafe_allow_html=True)
        for i, h in enumerate(headlines[:5]):
            dot = "🟢" if sent_val > 0 else "🔴"
            st.markdown(f"<p style='font-family:var(--font-mono);font-size:0.75rem;color:#90a4ae;border-bottom:1px solid #0d1520;padding:8px 0;margin:0'>{dot} {h}</p>", unsafe_allow_html=True)

    # ── Data Table ──
    with st.expander("📊 RAW DATA TABLE", expanded=False):
        display_cols = ["Date", "Close", "High", "Low", "Volume", "RSI", "MACD", "ATR", "BB_Pct", "sentiment", "Target"]
        show_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[show_cols].tail(30).style
                .background_gradient(subset=["Close"], cmap="YlOrRd")
                .format({"Close": "${:,.2f}", "RSI": "{:.1f}", "MACD": "{:.2f}", "ATR": "${:,.0f}", "BB_Pct": "{:.2%}", "sentiment": "{:+.3f}"}),
            use_container_width=True, height=300,
        )

    # ── Footer ──
    st.markdown("""
    <div class="terminal-footer">
        BITSCOPE INTELLIGENCE TERMINAL &nbsp;|&nbsp; JIIT 2026 &nbsp;|&nbsp;
        DATA: COINDESK + YFINANCE &nbsp;|&nbsp; MODEL: XGBOOST + VADER NLP &nbsp;|&nbsp;
        FOR EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
