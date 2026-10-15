
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
FORECAST_FILE = ROOT / "forecast_2025_final.npz"
GRAPH_DIR = ROOT / "ALL_GRAPHS"

st.set_page_config(
    page_title="IONOSPHERE LAB | TEC Forecasting",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(145deg, #07111f, #0b1424 55%, #101a2b);
    color: #e7eef8;
}
[data-testid="stSidebar"] {
    background: #091321;
    border-right: 1px solid #23334a;
}
[data-testid="stMetric"] {
    background: #111f32;
    border: 1px solid #243750;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #a8bbd2;
}
[data-testid="stMetricValue"] {
    color: #f3f7ff;
}
.block-container {
    padding-top: 2rem;
    max-width: 1600px;
}
.hero {
    background: linear-gradient(120deg, #11243a, #122b42);
    border: 1px solid #2a4662;
    border-radius: 18px;
    padding: 25px 28px;
    margin-bottom: 20px;
}
.hero-label {
    color: #7dd3fc;
    text-transform: uppercase;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    margin: 8px 0;
    color: #f4f8ff;
}
.hero-sub {
    color: #a9bdd3;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_forecasts(file_path, modified_time):
    with np.load(file_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def find_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def metrics_for(actual, predicted):
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    if actual.shape != predicted.shape:
        return None

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return None

    error = predicted - actual
    denominator = np.sum((actual - np.mean(actual)) ** 2)

    r2 = (
        1 - np.sum(error ** 2) / denominator
        if denominator > 0 else np.nan
    )

    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "R²": float(r2),
        "Bias": float(np.mean(error)),
        "Samples": int(len(actual)),
    }


def format_metric(value):
    return "N/A" if not np.isfinite(value) else f"{value:.3f}"


def flatten_for_plot(array):
    array = np.asarray(array, dtype=float)

    if array.ndim <= 1:
        return array.ravel()

    return np.nanmean(
        array.reshape(array.shape[0], -1),
        axis=1
    )


def daily_metrics(actual, predicted, dates):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if actual.shape != predicted.shape or actual.ndim == 0:
        return pd.DataFrame()

    if actual.shape[0] != len(dates):
        return pd.DataFrame()

    rows = []

    for i, date in enumerate(dates):
        result = metrics_for(actual[i], predicted[i])

        if result is not None:
            rows.append({
                "Date": pd.Timestamp(date),
                **result,
            })

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🌐 IONOSPHERE LAB")
st.sidebar.caption("TEC Forecasting & Scientific Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Scientific Overview",
        "Geomagnetic Regimes",
        "Forecast Explorer",
        "Scientific Graphs",
        "Error Analysis",
        "Methodology & Export",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Research configuration**")
st.sidebar.caption("Forecast period: 2025")
st.sidebar.caption("Models: LSTM · ConvLSTM")
st.sidebar.caption("Source: Saved forecast arrays")


# ============================================================
# LOAD SAVED FORECASTS
# ============================================================

if not FORECAST_FILE.exists():
    st.error(f"Forecast file not found: {FORECAST_FILE}")
    st.info(
        "Place forecast_2025_final.npz in the same folder as app.py."
    )
    st.stop()

try:
    raw = load_forecasts(
        str(FORECAST_FILE),
        FORECAST_FILE.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load forecast file: {exc}")
    st.stop()

conv_key = find_key(
    raw, ["conv_preds", "convlstm_preds", "conv_predictions"]
)
lstm_key = find_key(
    raw, ["lstm_preds", "lstm_predictions"]
)
actual_key = find_key(
    raw, ["actuals", "y_true", "ground_truth", "actual"]
)
doy_key = find_key(raw, ["doys", "day_of_year", "doy"])

if not all([conv_key, lstm_key, actual_key]):
    st.error("Required forecast arrays were not found.")
    st.write("Available array keys:", list(raw.keys()))
    st.stop()

conv = np.asarray(raw[conv_key], dtype=float)
lstm = np.asarray(raw[lstm_key], dtype=float)
actual = np.asarray(raw[actual_key], dtype=float)

if conv.shape != actual.shape or lstm.shape != actual.shape:
    st.error(
        f"Array shapes do not match. Actual: {actual.shape}; "
        f"LSTM: {lstm.shape}; ConvLSTM: {conv.shape}"
    )
    st.stop()

conv_metrics = metrics_for(actual, conv)
lstm_metrics = metrics_for(actual, lstm)

if conv_metrics is None or lstm_metrics is None:
    st.error("Unable to calculate metrics from the saved arrays.")
    st.stop()

# Use dates only when there is a valid date mapping.
n_samples = actual.shape[0] if actual.ndim > 0 else 1
dates = None
date_mapping_available = False

if doy_key:
    doy = np.asarray(raw[doy_key]).ravel()

    if len(doy) == n_samples:
        try:
            dates = (
                pd.to_datetime(
                    pd.DataFrame({
                        "year": np.full(n_samples, 2025),
                        "month": np.ones(n_samples),
                        "day": np.ones(n_samples),
                    })
                )
                + pd.to_timedelta(doy.astype(int) - 1, unit="D")
            )

            date_mapping_available = True

        except (ValueError, TypeError, OverflowError):
            dates = None

# Do not assume that 358 samples are 358 consecutive calendar days.
daily_aligned = (
    date_mapping_available
    and actual.ndim >= 2
    and len(dates) == actual.shape[0]
)

conv_series = flatten_for_plot(conv)
lstm_series = flatten_for_plot(lstm)
actual_series = flatten_for_plot(actual)

daily_conv = (
    daily_metrics(actual, conv, dates)
    if daily_aligned else pd.DataFrame()
)
daily_lstm = (
    daily_metrics(actual, lstm, dates)
    if daily_aligned else pd.DataFrame()
)


# ============================================================
# COMMON HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">
        Space Weather · Machine Learning · TEC
    </div>
    <div class="hero-title">IONOSPHERE LAB</div>
    <div class="hero-sub">
        Total Electron Content forecasting, model evaluation,
        and scientific analysis of observed and predicted TEC.
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Loaded: {FORECAST_FILE.name} · "
    f"Array shape: {actual.shape} · No model training performed"
)


# ============================================================
# PAGE 1: SCIENTIFIC OVERVIEW
# ============================================================

if page == "Scientific Overview":

    st.subheader("Scientific Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ConvLSTM RMSE",
        format_metric(conv_metrics["RMSE"]),
        "TECU",
    )
    c2.metric(
        "ConvLSTM MAE",
        format_metric(conv_metrics["MAE"]),
        "TECU",
    )
    c3.metric(
        "ConvLSTM R²",
        format_metric(conv_metrics["R²"]),
    )
    c4.metric(
        "Evaluated values",
        f'{conv_metrics["Samples"]:,}',
    )

    st.markdown("### Model performance comparison")

    comparison = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R²", "Bias"],
        "LSTM": [
            lstm_metrics["RMSE"],
            lstm_metrics["MAE"],
            lstm_metrics["R²"],
            lstm_metrics["Bias"],
        ],
        "ConvLSTM": [
            conv_metrics["RMSE"],
            conv_metrics["MAE"],
            conv_metrics["R²"],
            conv_metrics["Bias"],
        ],
    })

    st.dataframe(
        comparison.style.format({
            "LSTM": "{:.4f}",
            "ConvLSTM": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if lstm_metrics["RMSE"] > 0:
        improvement = (
            (lstm_metrics["RMSE"] - conv_metrics["RMSE"])
            / lstm_metrics["RMSE"] * 100
        )

        if improvement >= 0:
            st.success(
                f"ConvLSTM RMSE is {improvement:.1f}% lower "
                "than LSTM on these saved evaluation arrays."
            )
        else:
            st.info(
                f"ConvLSTM RMSE is {abs(improvement):.1f}% higher "
                "than LSTM on these saved evaluation arrays."
            )

    st.markdown("### Observed vs. predicted TEC")

    max_points = min(len(actual_series), 2500)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(max_points),
        y=actual_series[:max_points],
        name="Observed",
        line=dict(color="#FBBF24", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(max_points),
        y=lstm_series[:max_points],
        name="LSTM",
        line=dict(color="#A78BFA", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(max_points),
        y=conv_series[:max_points],
        name="ConvLSTM",
        line=dict(color="#38BDF8", width=1.8),
    ))

    fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_title="Sample index",
        yaxis_title="TEC (source units)",
        legend=dict(orientation="h", y=1.12),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scientific interpretation")

    if conv_metrics["RMSE"] < lstm_metrics["RMSE"]:
        st.write(
            "- ConvLSTM has lower aggregate RMSE on the loaded arrays.\n"
            "- MAE and bias should also be considered when interpreting "
            "forecast performance.\n"
            "- Quiet, normal, and severe performance must be evaluated "
            "separately before drawing conclusions about geomagnetic "
            "conditions."
        )
    else:
        st.write(
            "- LSTM has equal or lower aggregate RMSE on the loaded arrays.\n"
            "- Review MAE, bias, and regime-specific performance before "
            "selecting a preferred model."
        )


# ============================================================
# PAGE 2: GEOMAGNETIC REGIMES
# ============================================================

elif page == "Geomagnetic Regimes":

    st.subheader("Quiet, Normal & Severe Day Performance")

    st.write(
        "Upload independently sourced daily geomagnetic labels "
        "to compare forecast errors across regimes."
    )

    st.warning(
        "Regimes must be classified using geomagnetic observations, "
        "not by the model's forecast errors."
    )

    label_file = st.file_uploader(
        "Upload daily geomagnetic CSV",
        type=["csv"],
        help="CSV needs Date and either Regime or Kp columns.",
    )

    if not daily_aligned:
        st.error(
            "A valid date-of-year mapping is unavailable for the "
            "loaded forecast. Verify the dates before performing "
            "daily regime analysis."
        )

    elif label_file is None:
        st.info(
            "Required columns: Date and Regime, or Date and Kp."
        )

        st.code(
            "Date,Regime\n"
            "2025-01-01,Quiet\n"
            "2025-01-02,Normal\n"
            "2025-01-03,Severe",
            language="csv",
        )

        st.caption(
            "Example only. Do not use these illustrative labels "
            "as actual geomagnetic observations."
        )

    else:
        try:
            labels = pd.read_csv(label_file)
            labels.columns = [col.strip() for col in labels.columns]

            date_col = next(
                (
                    col for col in labels.columns
                    if col.lower() in ["date", "datetime"]
                ),
                None,
            )

            if date_col is None:
                st.error("CSV must contain a Date column.")

            else:
                labels["Date"] = pd.to_datetime(
                    labels[date_col],
                    errors="coerce",
                ).dt.normalize()

                regime_col = next(
                    (
                        col for col in labels.columns
                        if col.lower() == "regime"
                    ),
                    None,
                )

                kp_col = next(
                    (
                        col for col in labels.columns
                        if col.lower() == "kp"
                    ),
                    None,
                )

                if regime_col:
                    labels["Regime"] = (
                        labels[regime_col]
                        .astype(str)
                        .str.strip()
                        .str.title()
                    )

                elif kp_col:
                    labels[kp_col] = pd.to_numeric(
                        labels[kp_col],
                        errors="coerce",
                    )

                    st.caption(
                        "Default grouping used here: Quiet Kp ≤ 2; "
                        "Normal 2 < Kp < 5; Severe/Disturbed Kp ≥ 5. "
                        "Confirm the scientific classification with "
                        "your supervisor."
                    )

                    labels["Regime"] = np.select(
                        [
                            labels[kp_col] <= 2,
                            (labels[kp_col] > 2)
                            & (labels[kp_col] < 5),
                            labels[kp_col] >= 5,
                        ],
                        ["Quiet", "Normal", "Severe"],
                        default="Unknown",
                    )

                else:
                    st.error(
                        "CSV must contain either Regime or Kp."
                    )
                    st.stop()

                valid_regimes = ["Quiet", "Normal", "Severe"]

                labels = labels[
                    labels["Date"].notna()
                    & labels["Regime"].isin(valid_regimes)
                ].copy()

                conv_day = daily_conv.rename(columns={
                    "RMSE": "ConvLSTM RMSE",
                    "MAE": "ConvLSTM MAE",
                    "Bias": "ConvLSTM Bias",
                })

                lstm_day = daily_lstm.rename(columns={
                    "RMSE": "LSTM RMSE",
                    "MAE": "LSTM MAE",
                    "Bias": "LSTM Bias",
                })

                merged = (
                    labels[["Date", "Regime"]]
                    .merge(conv_day, on="Date", how="inner")
                    .merge(lstm_day, on="Date", how="inner")
                )

                if merged.empty:
                    st.error(
                        "No matching dates. Check the CSV dates "
                        "against the forecast date range."
                    )

                else:
                    st.caption(
                        f"Matched {merged['Date'].nunique()} labelled days."
                    )

                    summary = merged.groupby(
                        "Regime", observed=True
                    ).agg(
                        Days=("Date", "nunique"),
                        **{
                            "LSTM RMSE": ("LSTM RMSE", "mean"),
                            "ConvLSTM RMSE": ("ConvLSTM RMSE", "mean"),
                            "LSTM MAE": ("LSTM MAE", "mean"),
                            "ConvLSTM MAE": ("ConvLSTM MAE", "mean"),
                            "LSTM Bias": ("LSTM Bias", "mean"),
                            "ConvLSTM Bias": ("ConvLSTM Bias", "mean"),
                        },
                    ).reindex(valid_regimes)

                    st.markdown("### Regime-wise performance")

                    st.dataframe(
                        summary.style.format({
                            col: "{:.3f}"
                            for col in summary.columns
                            if col != "Days"
                        }),
                        use_container_width=True,
                    )

                    chart_df = summary.reset_index().melt(
                        id_vars="Regime",
                        value_vars=["LSTM RMSE", "ConvLSTM RMSE"],
                        var_name="Model",
                        value_name="Mean daily RMSE",
                    )

                    fig = px.bar(
                        chart_df,
                        x="Regime",
                        y="Mean daily RMSE",
                        color="Model",
                        barmode="group",
                        template="plotly_dark",
                        color_discrete_map={
                            "LSTM RMSE": "#A78BFA",
                            "ConvLSTM RMSE": "#38BDF8",
                        },
                    )

                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        merged.sort_values("Date"),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download regime analysis",
                        merged.to_csv(index=False).encode("utf-8"),
                        file_name="tec_regime_analysis_2025.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not process regime CSV: {exc}")


# ============================================================
# PAGE 3: FORECAST EXPLORER
# ============================================================

elif page == "Forecast Explorer":

    st.subheader("Forecast Explorer")

    if not daily_aligned:

        st.warning(
            "Daily date mapping is unavailable. Showing sample-level "
            "summaries instead of assuming that samples are calendar days."
        )

        max_n = min(len(actual_series), 2500)

        sample_df = pd.DataFrame({
            "Sample": np.arange(max_n),
            "Observed": actual_series[:max_n],
            "LSTM": lstm_series[:max_n],
            "ConvLSTM": conv_series[:max_n],
        })

        st.dataframe(
            sample_df.head(100),
            use_container_width=True,
        )

    else:

        selected_date = st.date_input(
            "Select forecast date",
            value=pd.Timestamp(dates.iloc[0]).date()
            if isinstance(dates, pd.Series)
            else pd.Timestamp(dates[0]).date(),
            min_value=pd.Timestamp(dates.min()).date(),
            max_value=pd.Timestamp(dates.max()).date(),
        )

        # FIX: Use .dt.date for Pandas Series.
        date_series = pd.to_datetime(
            pd.Series(dates),
            errors="coerce",
        )

        date_positions = np.where(
            date_series.dt.date.to_numpy() == selected_date
        )[0]

        if len(date_positions) == 0:

            st.warning("No saved forecast for this date.")

        else:

            idx = int(date_positions[0])

            y = np.asarray(actual[idx]).ravel()
            l = np.asarray(lstm[idx]).ravel()
            c = np.asarray(conv[idx]).ravel()

            max_n = min(len(y), 3000)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=np.arange(max_n),
                y=y[:max_n],
                name="Observed",
                line=dict(color="#FBBF24", width=2.3),
            ))

            fig.add_trace(go.Scatter(
                x=np.arange(max_n),
                y=l[:max_n],
                name="LSTM",
                line=dict(color="#A78BFA", width=1.6),
            ))

            fig.add_trace(go.Scatter(
                x=np.arange(max_n),
                y=c[:max_n],
                name="ConvLSTM",
                line=dict(color="#38BDF8", width=1.8),
            ))

            fig.update_layout(
                template="plotly_dark",
                height=420,
                xaxis_title="Grid point / value index",
                yaxis_title="TEC (source units)",
            )

            st.plotly_chart(fig, use_container_width=True)

            m1 = metrics_for(y, l)
            m2 = metrics_for(y, c)

            col1, col2, col3 = st.columns(3)

            col1.metric("LSTM RMSE", format_metric(m1["RMSE"]))
            col2.metric("ConvLSTM RMSE", format_metric(m2["RMSE"]))
            col3.metric("ConvLSTM MAE", format_metric(m2["MAE"]))

            st.markdown("### Residual comparison")

            residual_fig = go.Figure()

            residual_fig.add_trace(go.Scatter(
                x=np.arange(max_n),
                y=l[:max_n] - y[:max_n],
                name="LSTM residual",
                line=dict(color="#A78BFA"),
            ))

            residual_fig.add_trace(go.Scatter(
                x=np.arange(max_n),
                y=c[:max_n] - y[:max_n],
                name="ConvLSTM residual",
                line=dict(color="#38BDF8"),
            ))

            residual_fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="white",
            )

            residual_fig.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Grid point / value index",
                yaxis_title="Forecast - observed",
            )

            st.plotly_chart(
                residual_fig,
                use_container_width=True,
            )

            download_df = pd.DataFrame({
                "Observed": y,
                "LSTM": l,
                "ConvLSTM": c,
                "LSTM residual": l - y,
                "ConvLSTM residual": c - y,
            })

            st.download_button(
                "Download selected forecast data",
                download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"tec_forecast_{selected_date}.csv",
                mime="text/csv",
            )


# ============================================================
# PAGE 4: SCIENTIFIC GRAPHS
# ============================================================

elif page == "Scientific Graphs":

    st.subheader("Scientific Graph Gallery")

    if not GRAPH_DIR.exists():

        st.warning(f"Graph folder not found: {GRAPH_DIR}")

    else:

        image_files = sorted([
            path for path in GRAPH_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in [
                ".png", ".jpg", ".jpeg", ".webp"
            ]
        ])

        search = st.text_input(
            "Search graphs",
            placeholder="residual, comparison, monthly...",
        )

        if search.strip():
            image_files = [
                path for path in image_files
                if search.lower() in path.name.lower()
            ]

        if not image_files:

            st.info("No matching graph images found.")

        else:

            st.caption(f"{len(image_files)} graph(s) found")
            cols = st.columns(2)

            for i, img_path in enumerate(image_files):

                with cols[i % 2]:

                    with st.container(border=True):

                        st.markdown(
                            f"**{img_path.stem.replace('_', ' ')}**"
                        )

                        st.image(
                            str(img_path),
                            use_container_width=True,
                        )

                        st.download_button(
                            "Download figure",
                            data=img_path.read_bytes(),
                            file_name=img_path.name,
                            mime=(
                                "image/png"
                                if img_path.suffix.lower() == ".png"
                                else "image/jpeg"
                            ),
                            key=f"graph_{i}_{img_path.name}",
                        )


# ============================================================
# PAGE 5: ERROR ANALYSIS
# ============================================================

elif page == "Error Analysis":

    st.subheader("Forecast Error Analysis")

    a0 = actual.ravel()
    l0 = lstm.ravel()
    c0 = conv.ravel()

    valid = (
        np.isfinite(a0)
        & np.isfinite(l0)
        & np.isfinite(c0)
    )

    a = a0[valid]
    l = l0[valid]
    c = c0[valid]

    if len(a) == 0:

        st.error("No valid matching values for error analysis.")

    else:

        err_l = l - a
        err_c = c - a

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("LSTM mean error", f"{np.mean(err_l):.3f}")
        col2.metric("ConvLSTM mean error", f"{np.mean(err_c):.3f}")
        col3.metric("LSTM error std", f"{np.std(err_l):.3f}")
        col4.metric("ConvLSTM error std", f"{np.std(err_c):.3f}")

        tab1, tab2, tab3 = st.tabs([
            "Residual distribution",
            "Absolute error",
            "Daily trends",
        ])

        with tab1:

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=err_l,
                name="LSTM",
                opacity=0.65,
                nbinsx=60,
                marker_color="#A78BFA",
            ))

            fig.add_trace(go.Histogram(
                x=err_c,
                name="ConvLSTM",
                opacity=0.65,
                nbinsx=60,
                marker_color="#38BDF8",
            ))

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="white",
            )

            fig.update_layout(
                barmode="overlay",
                template="plotly_dark",
                height=420,
                xaxis_title="Forecast - observed",
                yaxis_title="Count",
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:

            abs_df = pd.DataFrame({
                "Model": ["LSTM", "ConvLSTM"],
                "Mean absolute error": [
                    np.mean(np.abs(err_l)),
                    np.mean(np.abs(err_c)),
                ],
                "Median absolute error": [
                    np.median(np.abs(err_l)),
                    np.median(np.abs(err_c)),
                ],
                "95th percentile absolute error": [
                    np.percentile(np.abs(err_l), 95),
                    np.percentile(np.abs(err_c), 95),
                ],
            })

            st.dataframe(
                abs_df,
                use_container_width=True,
                hide_index=True,
            )

            fig = go.Figure()

            fig.add_trace(go.Box(
                y=np.abs(err_l),
                name="LSTM",
                marker_color="#A78BFA",
                boxmean=True,
            ))

            fig.add_trace(go.Box(
                y=np.abs(err_c),
                name="ConvLSTM",
                marker_color="#38BDF8",
                boxmean=True,
            ))

            fig.update_layout(
                template="plotly_dark",
                height=400,
                yaxis_title="Absolute error",
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:

            if not daily_aligned or daily_conv.empty or daily_lstm.empty:

                st.info(
                    "Daily trends require a valid date mapping "
                    "and one sample per date."
                )

            else:

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=daily_lstm["Date"],
                    y=daily_lstm["RMSE"],
                    name="LSTM daily RMSE",
                    line=dict(color="#A78BFA"),
                ))

                fig.add_trace(go.Scatter(
                    x=daily_conv["Date"],
                    y=daily_conv["RMSE"],
                    name="ConvLSTM daily RMSE",
                    line=dict(color="#38BDF8"),
                ))

                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Daily RMSE",
                )

                st.plotly_chart(fig, use_container_width=True)

                daily_compare = daily_lstm[
                    ["Date", "RMSE", "MAE", "Bias"]
                ].merge(
                    daily_conv[["Date", "RMSE", "MAE", "Bias"]],
                    on="Date",
                    suffixes=("_LSTM", "_ConvLSTM"),
                )

                st.dataframe(
                    daily_compare,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download daily error metrics",
                    daily_compare.to_csv(index=False).encode("utf-8"),
                    file_name="tec_daily_error_metrics.csv",
                    mime="text/csv",
                )


# ============================================================
# PAGE 6: METHODOLOGY & EXPORT
# ============================================================

elif page == "Methodology & Export":

    st.subheader("Methodology & Reproducibility")

    st.write(
        "This dashboard loads saved LSTM and ConvLSTM predictions. "
        "It does not retrain either model."
    )

    st.markdown("### Loaded arrays")

    array_summary = pd.DataFrame([
        {
            "Array": "Observed TEC",
            "NPZ key": actual_key,
            "Shape": str(actual.shape),
            "Values": actual.size,
        },
        {
            "Array": "LSTM prediction",
            "NPZ key": lstm_key,
            "Shape": str(lstm.shape),
            "Values": lstm.size,
        },
        {
            "Array": "ConvLSTM prediction",
            "NPZ key": conv_key,
            "Shape": str(conv.shape),
            "Values": conv.size,
        },
    ])

    st.dataframe(
        array_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Evaluation metrics")

    st.write("""
    - **RMSE:** penalizes larger errors more strongly.
    - **MAE:** average absolute difference between forecast and observation.
    - **R²:** coefficient of determination; not forecast accuracy.
    - **Bias:** mean forecast minus observation; positive bias indicates
      average overprediction.
    """)

    st.markdown("### Reporting checks")

    st.checkbox(
        "Observed and predicted arrays use matching units.",
        key="check_units",
    )

    st.checkbox(
        "Forecast dates align with observations.",
        key="check_dates",
    )

    st.checkbox(
        "Geomagnetic labels come from an independent source.",
        key="check_regimes",
    )

    export_metrics = pd.DataFrame([
        {"Model": "LSTM", **lstm_metrics},
        {"Model": "ConvLSTM", **conv_metrics},
    ])

    st.download_button(
        "Download overall model metrics",
        export_metrics.to_csv(index=False).encode("utf-8"),
        file_name="tec_model_metrics_2025.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download forecast data",
        pd.DataFrame({
            "Observed": actual.ravel(),
            "LSTM": lstm.ravel(),
            "ConvLSTM": conv.ravel(),
        }).to_csv(index=False).encode("utf-8"),
        file_name="tec_forecasts_2025.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("IONOSPHERE LAB · Scientific analysis interface")
