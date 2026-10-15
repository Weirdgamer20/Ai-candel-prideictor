import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import data_quality_report, fetch_candles, save_candles
from src.modeling import fit_and_evaluate, load_models, predict_next
from src.ocr_utils import extract_chart_data
from src.universe import discover_spot_markets


st.set_page_config(
    page_title="AI Crypto Candle Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def cached_spot_markets(quote_filter: str):
    return discover_spot_markets(None if quote_filter == "All quotes" else (quote_filter,))


def safe_market_slug(asset: str, quote: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", f"{asset}_{quote}".lower()).strip("_")


def infer_assets_from_pair(pair: str) -> tuple[str, str]:
    pair_body = pair.split("-", 1)[-1]
    parts = pair_body.split("_", 1)
    if len(parts) == 2 and all(parts):
        return parts[0].upper(), parts[1].upper()
    return "UNKNOWN", "UNKNOWN"

st.title("AI Crypto Candle Predictor")
st.caption("CoinDCX spot-market research · Multi-asset candle forecasts · Risk-gated decision foundation · OCR verification")

st.info(
    "**Boundary Notice**: Research & prediction tool only. Predictions are probabilistic model estimates and "
    "never guarantee market outcomes. CoinDCX holds exchange custody; this system does not store API secrets, "
    "read account balances, or execute live orders.",
    icon="ℹ️",
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Market Data Engine")
    quote_filter = st.selectbox("Quote currency", ["INR", "USDT", "All quotes"], index=0)
    try:
        active_markets = cached_spot_markets(quote_filter)
    except Exception as exc:
        active_markets = []
        st.warning(f"Could not refresh CoinDCX market list: {type(exc).__name__}: {exc}")
    pair_options = [market.pair for market in active_markets]
    preferred_pair = next(
        (i for i, market in enumerate(active_markets)
         if market.base_asset == "BTC" and market.quote_asset == "INR"),
        0,
    )
    if pair_options:
        selected_pair = st.selectbox("Active CoinDCX spot pair", pair_options, index=preferred_pair)
        selected_market = next(market for market in active_markets if market.pair == selected_pair)
        selected_asset, selected_quote = selected_market.base_asset, selected_market.quote_asset
    else:
        selected_pair = st.text_input("CoinDCX pair ID (manual fallback)", value="I-BTC_INR")
        selected_asset, selected_quote = infer_assets_from_pair(selected_pair)

    market_slug = safe_market_slug(selected_asset, selected_quote)
    # Preserve the original BTC/INR data/model locations for backward compatibility.
    if selected_asset == "BTC" and selected_quote == "INR":
        DATA_PATH = Path("data/raw/btc_inr_15m.csv")
        MODEL_DIR = Path("models")
    else:
        DATA_PATH = Path("data/raw") / f"{market_slug}_15m.csv"
        MODEL_DIR = Path("models") / market_slug
    currency_prefix = "₹" if selected_quote == "INR" else f"{selected_quote} "

    start_date_input = st.date_input(
        "Historical Target Start",
        value=pd.to_datetime("2018-10-18"),
        help="CoinDCX's earliest available BTC/INR candle is 2018-10-18. Earlier dates return empty sets.",
    )
    force_fresh = st.checkbox("Download from scratch (no resumption)", value=False)

    if st.button("📥 Download / Refresh Candles", type="primary", use_container_width=True):
        try:
            with st.spinner("Connecting to CoinDCX and ingesting 15m candles..."):
                pair = selected_pair.strip()
                resume_file = None if force_fresh else str(DATA_PATH)
                candles = fetch_candles(
                    pair=pair,
                    start_date=str(start_date_input),
                    resume_path=resume_file,
                )
                save_candles(candles, str(DATA_PATH))
                is_resumed = candles.attrs.get("resumed", False)
                mode_str = "Incrementally updated" if is_resumed else "Downloaded"
                st.session_state["data_message"] = f"✅ {mode_str} {len(candles):,} candles for {pair}."
                st.session_state.pop("data_error", None)
        except Exception as exc:
            st.session_state["data_error"] = f"Failed to ingest: {type(exc).__name__}: {exc}"

    if st.session_state.get("data_message"):
        st.success(st.session_state["data_message"])
    if st.session_state.get("data_error"):
        st.error(st.session_state["data_error"])

    st.divider()
    st.markdown("### 💾 Local State")
    if DATA_PATH.exists():
        st.caption(f"Local storage for {selected_pair}: `{DATA_PATH}` ({DATA_PATH.stat().st_size / (1024*1024):.2f} MB)")
    if (MODEL_DIR / "direction_model.joblib").exists():
        st.caption(f"Trained models available for {selected_pair} in `{MODEL_DIR}`")

# ----------------- DATA LOADING -----------------
if not DATA_PATH.exists():
    st.warning("No candle dataset found locally. Use **Download / Refresh Candles** in the sidebar to ingest history.")
    st.code("Use the active-market selector to download the selected pair, then train its models in the Forecast tab.", language="text")
    st.stop()

raw = pd.read_csv(DATA_PATH, parse_dates=["time"])
required = {"time", "open", "high", "low", "close", "volume"}
if not required.issubset(raw.columns):
    st.error(f"Dataset corrupted or missing required columns: {sorted(required - set(raw.columns))}")
    st.stop()

raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
for col in ["open", "high", "low", "close", "volume"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw = raw.dropna(subset=list(required)).drop_duplicates("time").sort_values("time").reset_index(drop=True)
if len(raw) < 100:
    st.error(f"Dataset too small ({len(raw)} candles). Minimum 100 rows required.")
    st.stop()

quality = data_quality_report(raw)

# ----------------- TOP METRICS BANNER -----------------
latest_close = raw["close"].iloc[-1]
first_close = raw["close"].iloc[0]
change_24h_candles = min(96, len(raw)) # 96 * 15m = 24h
day_change_pct = ((latest_close / raw["close"].iloc[-change_24h_candles]) - 1) * 100

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"Latest {selected_asset}/{selected_quote}", f"{currency_prefix}{latest_close:,.8g}", f"{day_change_pct:+.2f}% over the last 24h")
m2.metric("Total Candles", f"{len(raw):,}")
m3.metric("Coverage in Span", f"{quality.get('observed_span_coverage_pct', 0.0):.1f}%")
m4.metric("Earliest Candle", str(raw["time"].min())[:16])
m5.metric("Latest Candle", str(raw["time"].max())[:16])

# ----------------- TABS -----------------
tab_market, tab_predict, tab_eval, tab_ocr, tab_data = st.tabs([
    "📈 Market Overview",
    "🔮 Next-Candle Forecast",
    "📊 Model Evaluation",
    "🔍 OCR Chart Inspector",
    "📋 Data Quality & Export",
])

# 1. MARKET OVERVIEW
with tab_market:
    st.subheader("BTC/INR Spot Price Chart (15-Minute Candles)")
    chart_col1, chart_col2 = st.columns([1, 3])
    with chart_col1:
        chart_mode = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        display_n = st.slider("Visible candles", min_value=50, max_value=min(3000, len(raw)), value=min(300, len(raw)), step=50)

    sub_df = raw.tail(display_n)
    fig = go.Figure()
    if chart_mode == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=sub_df["time"], open=sub_df["open"], high=sub_df["high"],
            low=sub_df["low"], close=sub_df["close"], name="BTC/INR 15m"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=sub_df["time"], y=sub_df["close"], mode="lines", name="Close",
            line=dict(color="#f7931a", width=2)
        ))
    fig.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Timestamp (IST)",
        yaxis_title=f"Price ({selected_quote})",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. NEXT-CANDLE FORECAST
with tab_predict:
    st.subheader("Next 15-Minute Candle Prediction")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        train_trigger = st.button("🚀 Train & Evaluate Models (Chronological)", type="primary", use_container_width=True)
    with btn_col2:
        predict_trigger = st.button("⚡ Generate Forecast with Saved Models", use_container_width=True)

    if train_trigger:
        try:
            with st.spinner("Training baseline, Random Forest, and Gradient Boosting models..."):
                metrics, test_res, clf, reg = fit_and_evaluate(raw, str(MODEL_DIR))
                pred = predict_next(raw, clf, reg)
                st.session_state["model_metrics"] = metrics
                st.session_state["test_results"] = test_res
                st.session_state["next_prediction"] = pred
                st.success(f"Models successfully trained and saved! Best classifier: **{metrics.get('selected_classifier')}**")
        except Exception as exc:
            st.error(f"Training failed: {type(exc).__name__}: {exc}")

    if predict_trigger:
        try:
            clf, reg = load_models(str(MODEL_DIR))
            pred = predict_next(raw, clf, reg)
            st.session_state["next_prediction"] = pred
            st.success("Loaded saved models and generated forecast.")
        except Exception as exc:
            st.error(f"Failed to generate prediction: {type(exc).__name__}: {exc}")

    pred = st.session_state.get("next_prediction")
    if pred:
        st.markdown("---")
        p1, p2, p3, p4 = st.columns(4)
        dir_color = "🟢" if pred["direction"] == "UP" else "🔴"
        p1.metric("Predicted Direction", f"{dir_color} {pred['direction']}", f"{pred['probability_up']:.1%} probability UP")
        p2.metric("Estimated Next Close", f"{currency_prefix}{pred['predicted_next_close']:,.8g}", f"{pred['predicted_change_pct']:+.3f}%")
        p3.metric("Expected Price Move", f"{currency_prefix}{pred['predicted_change_inr']:+,.6g}")
        p4.metric("Last Completed Close", f"{currency_prefix}{pred['last_close']:,.8g}")

        st.caption(
            f"📅 Base candle: **{pred['candle_time']}** | Forecast Horizon: **Next 15 minutes** | "
            f"Generated at: `{pred.get('prediction_timestamp_utc', 'N/A')}`"
        )

        # Visual forecast bridge
        bridge_df = raw.tail(80)
        f_fig = go.Figure()
        f_fig.add_trace(go.Scatter(
            x=bridge_df["time"], y=bridge_df["close"], mode="lines+markers",
            name="Observed Closes", line=dict(color="#00d2be", width=2)
        ))
        f_fig.add_trace(go.Scatter(
            x=[bridge_df["time"].iloc[-1], bridge_df["time"].iloc[-1] + pd.Timedelta(minutes=15)],
            y=[bridge_df["close"].iloc[-1], pred["predicted_next_close"]],
            mode="lines+markers", name="Forecast", line=dict(color="#f39c12", width=3, dash="dash"),
            marker=dict(size=[6, 12], color="#f39c12", symbol=["circle", "star"])
        ))
        f_fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Time",
            yaxis_title=f"{selected_asset}/{selected_quote} price",
            template="plotly_dark",
        )
        st.plotly_chart(f_fig, use_container_width=True)

# 3. MODEL EVALUATION
with tab_eval:
    st.subheader("Strict Chronological Out-of-Sample Holdout")
    metrics = st.session_state.get("model_metrics")
    if not metrics and (MODEL_DIR / "metrics.json").exists():
        try:
            metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    if metrics:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Selected Classifier", metrics.get("selected_classifier", "N/A"))
        e2.metric("Test Direction Accuracy", f"{metrics.get('test_accuracy', 0):.2%}", f"Baseline: {metrics.get('test_majority_direction_baseline_accuracy', 0):.2%}")
        e3.metric("Test Close MAE", f"₹{metrics.get('test_close_mae_inr', 0):,.2f}", f"Naive: ₹{metrics.get('naive_close_baseline_mae_inr', 0):,.2f}")
        e4.metric("Close MAPE", f"{metrics.get('test_close_mape_pct', 0):.2f}%", f"Naive: {metrics.get('naive_close_baseline_mape_pct', 0):.2f}%")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Classification Metrics")
            clf_df = pd.DataFrame({
                "Metric": ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Majority Baseline"],
                "Value": [
                    f"{metrics.get('test_accuracy', 0):.4f}",
                    f"{metrics.get('test_balanced_accuracy', 0):.4f}",
                    f"{metrics.get('test_precision', 0):.4f}",
                    f"{metrics.get('test_recall', 0):.4f}",
                    f"{metrics.get('test_f1', 0):.4f}",
                    f"{metrics.get('test_roc_auc') or 0:.4f}",
                    f"{metrics.get('test_majority_direction_baseline_accuracy', 0):.4f}",
                ]
            })
            st.table(clf_df)

            cm = metrics.get("test_confusion_matrix")
            if cm and len(cm) == 2:
                st.markdown("#### Confusion Matrix")
                cm_df = pd.DataFrame(cm, index=["Actual DOWN", "Actual UP"], columns=["Pred DOWN", "Pred UP"])
                st.table(cm_df)

        with col_r:
            st.markdown("#### Regression Error Comparison")
            reg_df = pd.DataFrame({
                "Model": ["Machine Learning (HistGB)", "Naive Benchmark (Next = Current)"],
                f"MAE ({selected_quote})": [f"{currency_prefix}{metrics.get('test_close_mae_inr', 0):,.6g}", f"{currency_prefix}{metrics.get('naive_close_baseline_mae_inr', 0):,.6g}"],
                f"RMSE ({selected_quote})": [f"{currency_prefix}{metrics.get('test_close_rmse_inr', 0):,.6g}", f"{currency_prefix}{metrics.get('naive_close_baseline_rmse_inr', 0):,.6g}"],
                "MAPE (%)": [f"{metrics.get('test_close_mape_pct', 0):.2f}%", f"{metrics.get('naive_close_baseline_mape_pct', 0):.2f}%"],
            })
            st.table(reg_df)

            st.markdown("#### Validation Benchmark Scores")
            st.json(metrics.get("validation_balanced_accuracy_by_model", {}))

        test_pred_path = MODEL_DIR / "test_predictions.csv"
        if test_pred_path.exists():
            tdf = pd.read_csv(test_pred_path)
            st.markdown("#### Out-of-Sample Predictions vs Actual Outcomes")
            st.dataframe(tdf.tail(100), use_container_width=True)
            st.download_button(
                "📥 Download Out-of-Sample Predictions CSV",
                tdf.to_csv(index=False),
                file_name="test_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("No trained models found. Train models under the **Next-Candle Forecast** tab to view out-of-sample evaluation.")

# 4. OCR CHART INSPECTOR
with tab_ocr:
    st.subheader("Chart Screenshot OCR & Price Extraction")
    st.markdown(
        "Upload a chart screenshot to extract visible price labels and timestamps. "
        "**Safety Policy**: OCR results are display-only and cannot silently contaminate or alter the exchange candle dataset."
    )

    uploaded_img = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_img is not None:
        col_img, col_out = st.columns([1, 1])
        with col_img:
            st.image(uploaded_img, caption="Target Screenshot", use_container_width=True)

        with col_out:
            if st.button("🔍 Extract Text & Detect Prices", type="primary", use_container_width=True):
                with st.spinner("Processing image and running Tesseract OCR..."):
                    try:
                        ocr_data = extract_chart_data(uploaded_img)
                        st.session_state["ocr_data"] = ocr_data
                    except Exception as exc:
                        st.error(f"OCR failed: {type(exc).__name__}: {exc}")

            ocr_res = st.session_state.get("ocr_data")
            if ocr_res:
                st.metric("OCR Mean Confidence", f"{ocr_res.get('confidence_pct', 0)}%")
                if ocr_res.get("parsed_prices"):
                    st.markdown("**Detected Candidate Prices:**")
                    for p in ocr_res["parsed_prices"]:
                        st.write(f"- `{p['raw']}` → **₹{p['value']:,.2f}**")
                if ocr_res.get("parsed_timestamps"):
                    st.markdown(f"**Detected Timestamps:** {', '.join(ocr_res['parsed_timestamps'])}")

                user_correction = st.text_area(
                    "Manual Inspection & Correction Sandbox (Display Only)",
                    value=ocr_res.get("raw_text", ""),
                    height=180,
                )
                st.caption("Editing here does not write to historical market databases.")

# 5. DATA QUALITY & EXPORT
with tab_data:
    st.subheader("Dataset Integrity & Coverage Diagnostics")
    st.json(quality)

    st.markdown("#### Recent 100 Ingested Candles")
    st.dataframe(raw.tail(100), use_container_width=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            "📥 Export Full Ingested Candles (CSV)",
            raw.to_csv(index=False),
            file_name=f"{market_slug}_15m.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d_col2:
        st.download_button(
            "📥 Export Quality Report (JSON)",
            json.dumps(quality, indent=2),
            file_name="data_quality_report.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()
st.caption(
    f"{selected_asset}/{selected_quote} Candle Predictor · Built for local execution on Windows 11 / WSL · "
    "Chronological holdout validation · Deterministic feature pipeline"
)
