import streamlit as st
import os
os.environ['YFINANCE_DISABLE_CACHE'] = 'True'  # yfinance 캐시 비활성화 (포럼 팁)
try:
    import yfinance as yf
except ImportError:
    st.error("yfinance 설치 실패. requirements.txt 확인하세요.")
    st.stop()
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="옵션 분석기 v3", layout="centered")
st.title("미국 주식 옵션 달성 가능성 분석기 v3")
st.caption("by @joseunghye42513 (istroboy-dot) | Black-Scholes + Monte Carlo")

with st.sidebar:
    st.header("입력 정보")
    ticker = st.text_input("종목 티커 (예: AAPL, TSLA)", "AAPL").upper()
    target_strike = st.number_input("목표 행사가", min_value=0.01, value=200.0)
    option_type = st.selectbox("옵션 타입", ["call", "put"])
    manual_expiry = st.text_input("만기일 (YYYY-MM-DD, 비워두면 자동)", "")
    simulations = st.slider("시뮬레이션 수", 100, 1000, 500, 100)

@st.cache_data(ttl=300)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError(f"{ticker} 데이터 없음")
        price = hist['Close'].iloc[-1]
        expirations = stock.options
        if not expirations:
            raise ValueError(f"{ticker} 옵션 데이터 없음")
        return price, expirations, stock
    except Exception as e:
        st.error(f"오류: {str(e)}. 티커 확인하세요.")
        raise

# (나머지 함수: black_scholes_call, black_scholes_put, monte_carlo_simulation – 이전 코드와 동일)
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(-d1)

def monte_carlo_simulation(S, K, T, r, sigma, option_type, simulations=500):
    if T <= 0: return np.zeros((1,1)), 0.5
    dt = 1/365; N = max(int(T * 365), 1)
    paths = np.zeros((simulations, N+1)); paths[:, 0] = S
    for t in range(1, N+1):
        z = np.random.standard_normal(simulations)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    final = paths[:, -1]
    itm = (final > K).astype(float) if option_type == "call" else (final < K).astype(float)
    return paths, np.mean(itm)

if st.button("분석 시작", type="primary"):
    with st.spinner("데이터 로딩 중..."):
        try:
            current_price, expirations, stock = get_market_data(ticker)
        except:
            st.stop()

    expiry = manual_expiry if manual_expiry and manual_expiry in expirations else expirations[0]
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    days_left = (expiry_dt - datetime.now()).days
    T = max(days_left / 365.0, 0.001)

    opt = stock.option_chain(expiry)
    chain = opt.calls if option_type == "call" else opt.puts
    if chain.empty: st.error("옵션 데이터 없음"); st.stop()
    near_strike = chain.iloc[(chain['strike'] - target_strike).abs().argsort()[:1]]
    iv = near_strike['impliedVolatility'].iloc[0] if not near_strike.empty else chain['impliedVolatility'].mean()
    iv = max(iv, 0.05); r = 0.04

    prob_delta = black_scholes_call(current_price, target_strike, T, r, iv) if option_type == "call" else black_scholes_put(current_price, target_strike, T, r, iv)

    with st.spinner("시뮬레이션 중..."):
        paths, prob_mc = monte_carlo_simulation(current_price, target_strike, T, r, iv, option_type, simulations)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("현재 주가", f"${current_price:.2f}")
        st.metric("행사가", f"${target_strike:.2f}")
        st.metric("남은 일수", f"{days_left}일")
    with col2:
        st.metric("IV", f"{iv*100:.2f}%")
        st.metric("Delta 확률", f"{prob_delta*100:.2f}%")
        st.metric("MC 확률", f"{prob_mc*100:.2f}%")

    fig = go.Figure()
    for i in range(min(50, simulations)):
        fig.add_trace(go.Scatter(y=paths[i], mode='lines', line=dict(color='lightgray'), showlegend=False))
    fig.add_hline(y=target_strike, line_dash="dash", line_color="red")
    fig.add_hline(y=current_price, line_color="blue")
    fig.update_layout(title=f"{ticker} {option_type.upper()} 시뮬레이션", height=400)
    st.plotly_chart(fig)

    st.success(f"달성 확률: {prob_delta*100:.1f}%")

else:
    st.info("입력 후 버튼 클릭!")
