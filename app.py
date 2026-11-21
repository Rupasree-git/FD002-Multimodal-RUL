# app.py
# =========================================================
# NASA C-MAPSS FD002 — Prognostics & TrustGate Console
# =========================================================

from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_fd002_data
from src.inference import load_artifacts, predict_trajectories
from src.preprocessing import compute_rul_full

st.set_page_config(
    page_title="C-MAPSS FD002 | Multimodal RUL & TrustGate Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
        .console-eyebrow {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        .console-title {
            font-size: 1.65rem;
            font-weight: 650;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin: 0 0 0.4rem 0;
        }
        .console-subtitle {
            font-size: 0.92rem;
            color: #64748b;
            margin: 0 0 1.25rem 0;
        }
        .section-heading {
            font-size: 1.02rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            margin: 0.25rem 0 0.65rem 0;
        }
        .status-alert {
            padding: 0.55rem 0.8rem;
            border-left: 3px solid #b91c1c;
            background-color: rgba(185, 28, 28, 0.06);
            font-size: 0.88rem;
            font-weight: 500;
        }
        .status-ok {
            padding: 0.55rem 0.8rem;
            border-left: 3px solid #15803d;
            background-color: rgba(21, 128, 61, 0.06);
            font-size: 0.88rem;
            font-weight: 500;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts(models_dir: str) -> dict:
    return load_artifacts(models_dir=models_dir, device="cpu")


@st.cache_data(show_spinner="Loading FD002 dataset...")
def get_dataset(data_dir: str):
    return load_fd002_data(data_dir)


def _render_rul_chart(preds_df: pd.DataFrame) -> alt.Chart:
    value_cols = [
        c
        for c in [
            "pred_rul_trustgate",
            "pred_rul_sensor",
            "pred_rul_visual",
            "pred_rul_tabular",
            "true_rul",
        ]
        if c in preds_df.columns
    ]
    label_map = {
        "pred_rul_trustgate": "TrustGate Fused",
        "pred_rul_sensor": "Sensor RF",
        "pred_rul_visual": "Visual RF",
        "pred_rul_tabular": "Tabular RF",
        "true_rul": "Ground Truth RUL",
    }
    melted = preds_df[["cycle"] + value_cols].melt(
        id_vars="cycle", var_name="series", value_name="rul_cycles"
    )
    melted["Series"] = melted["series"].map(label_map).fillna(melted["series"])

    return (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("cycle:Q", title="Operating Cycle"),
            y=alt.Y("rul_cycles:Q", title="Remaining Useful Life (cycles)"),
            color=alt.Color("Series:N", title="Model / Reference"),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("rul_cycles:Q", title="RUL (cycles)", format=".1f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )


def _render_weights_chart(preds_df: pd.DataFrame) -> alt.Chart:
    weight_cols = [c for c in preds_df.columns if c.startswith("trust_weight_")]
    label_map = {
        "trust_weight_sensor": "Sensor",
        "trust_weight_visual": "Visual Proxy",
        "trust_weight_tabular": "Tabular Context",
    }
    melted = preds_df[["cycle"] + weight_cols].melt(
        id_vars="cycle", var_name="modality", value_name="weight"
    )
    melted["Modality"] = melted["modality"].map(label_map).fillna(melted["modality"])

    return (
        alt.Chart(melted)
        .mark_area(opacity=0.78)
        .encode(
            x=alt.X("cycle:Q", title="Operating Cycle"),
            y=alt.Y(
                "weight:Q",
                stack="normalize",
                title="Normalized Trust Allocation",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("Modality:N", title="Modality"),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("Modality:N", title="Modality"),
                alt.Tooltip("weight:Q", title="Weight", format=".3f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )


def _render_disagreement_chart(preds_df: pd.DataFrame, threshold: float) -> alt.Chart:
    base = alt.Chart(preds_df).encode(x=alt.X("cycle:Q", title="Operating Cycle"))

    std_line = base.mark_line(color="#1d4ed8", strokeWidth=2).encode(
        y=alt.Y(
            "modality_disagreement_std:Q",
            title="Cross-Modality Std Dev (cycles)",
        ),
        tooltip=[
            alt.Tooltip("cycle:Q", title="Cycle"),
            alt.Tooltip(
                "modality_disagreement_std:Q",
                title="Modality Std (cycles)",
                format=".2f",
            ),
            alt.Tooltip("failure_risk:Q", title="Failure Risk", format=".3f"),
        ],
    )

    thresh_df = pd.DataFrame({"threshold": [threshold]})
    thresh_rule = (
        alt.Chart(thresh_df)
        .mark_rule(color="#b91c1c", strokeDash=[5, 4], strokeWidth=1.5)
        .encode(y="threshold:Q")
    )

    return (std_line + thresh_rule).properties(height=240).interactive()


def main() -> None:
    st.markdown('<div class="console-eyebrow">Turbofan Prognostics &amp; Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="console-title">NASA C-MAPSS FD002 — Trust-Aware Multimodal RUL</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="console-subtitle">Engine-disjoint multimodal RUL inference, TrustGate dynamic modality weighting, '
        "exponential failure risk scoring, and inter-modality contradiction screening.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Configuration")
        models_dir = st.text_input("Artifacts Directory", value="models")
        data_dir = st.text_input("Dataset Directory", value="data")

        uploaded_file = st.file_uploader("Custom Trajectory CSV", type=["csv"])
        split_choice = "test"
        if uploaded_file is None:
            split_choice = st.radio(
                "FD002 Split",
                options=["test", "train"],
                index=0,
                help="Select official truncated test set (test_FD002.txt) or run-to-failure training set (train_FD002.txt).",
            )

    try:
        artifacts = get_artifacts(models_dir)
    except Exception as exc:
        st.error(f"Failed to load artifacts from '{models_dir}': {exc}")
        st.caption(
            "Expected artifacts: pipeline_metadata.json, pca_visual.joblib, "
            "rf_<modality>.joblib, trustgate.pt, and trustgate_scaler.joblib."
        )
        return

    default_threshold = float(artifacts["metadata"].get("contradiction_threshold", 50.0))
    threshold = st.sidebar.slider(
        "Contradiction Threshold (cycles)",
        min_value=10.0,
        max_value=120.0,
        value=default_threshold,
        step=5.0,
        help="Flags cycles where standard deviation across modality RUL predictions exceeds this threshold.",
    )

    try:
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            has_gt = "RUL" in raw_df.columns
            source_label = f"Uploaded CSV: {uploaded_file.name}"
        else:
            train_df, test_df, _ = get_dataset(data_dir)
            if split_choice == "train":
                raw_df = compute_rul_full(train_df)
                has_gt = True
                source_label = "FD002 Training Set (train_FD002.txt)"
            else:
                raw_df = test_df
                has_gt = False
                source_label = "FD002 Test Set (test_FD002.txt)"
    except Exception as exc:
        st.error(f"Failed to load trajectory data from '{data_dir}': {exc}")
        st.caption("Place train_FD002.txt and test_FD002.txt in the dataset directory.")
        return

    if "engine_id" not in raw_df.columns or raw_df.empty:
        st.error("Trajectory data must contain a non-empty 'engine_id' column.")
        return

    engine_ids = sorted(raw_df["engine_id"].unique().tolist())

    ctrl_left, ctrl_right = st.columns([1, 2])
    with ctrl_left:
        selected_engine = st.selectbox("Engine ID", options=engine_ids, index=0)
    with ctrl_right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"Source: {source_label} | Observed cycles: {int((raw_df['engine_id'] == selected_engine).sum())}")

    engine_df = raw_df[raw_df["engine_id"] == selected_engine].copy()

    try:
        preds_df, summary = predict_trajectories(
            engine_df,
            artifacts=artifacts,
            contradiction_threshold=threshold,
            has_ground_truth_rul=has_gt,
        )
    except Exception as exc:
        st.error(f"Inference error for engine {selected_engine}: {exc}")
        return

    latest = preds_df.iloc[-1]
    is_contradiction = bool(latest["contradiction_flag"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TrustGate RUL (Latest)", f"{latest['pred_rul_trustgate']:.1f} cycles")
    m2.metric("Failure Risk Score", f"{latest['failure_risk']:.3f}")
    m3.metric("Modality Std Dev", f"{latest['modality_disagreement_std']:.1f} cycles")
    m4.metric(
        "Flagged Cycles",
        f"{int(preds_df['contradiction_flag'].sum())} / {len(preds_df)}",
    )

    if is_contradiction:
        st.markdown(
            f'<div class="status-alert">Contradiction flagged at latest cycle ({int(latest["cycle"])}): '
            f'modality spread ({latest["modality_disagreement_std"]:.1f} cycles) exceeds '
            f'threshold ({threshold:.0f} cycles). Secondary review recommended.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-ok">Modalities aligned at latest cycle ({int(latest["cycle"])}): '
            f'modality spread ({latest["modality_disagreement_std"]:.1f} cycles) is within '
            f'threshold ({threshold:.0f} cycles).</div>',
            unsafe_allow_html=True,
        )

    if "true_rul" in latest and not pd.isna(latest["true_rul"]):
        st.caption(
            f"Latest ground-truth RUL: {latest['true_rul']:.1f} cycles | "
            f"Trajectory MAE: {summary.get('mae', np.nan):.2f} cycles | "
            f"NASA Score: {summary.get('nasa_score', np.nan):,.0f}"
        )

    st.markdown("---")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.markdown('<div class="section-heading">RUL Trajectory by Modality</div>', unsafe_allow_html=True)
        st.altair_chart(_render_rul_chart(preds_df), use_container_width=True)

    with chart_col_2:
        st.markdown('<div class="section-heading">TrustGate Modality Weights</div>', unsafe_allow_html=True)
        st.altair_chart(_render_weights_chart(preds_df), use_container_width=True)

    st.markdown(
        '<div class="section-heading">Inter-Modality Disagreement Against Threshold</div>',
        unsafe_allow_html=True,
    )
    st.altair_chart(_render_disagreement_chart(preds_df, threshold), use_container_width=True)

    st.markdown('<div class="section-heading">Cycle-Level Inference Log</div>', unsafe_allow_html=True)
    with st.expander("Inspect trajectory predictions table", expanded=False):
        st.dataframe(preds_df.round(3), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()