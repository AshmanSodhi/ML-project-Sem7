"""LLM-Assisted Mood Detection — Live Demo Dashboard (PRD S18).
Run: streamlit run app/dashboard.py
Uses ONLY behavioural metadata. Typed content is NEVER collected.
"""
import json
import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keystroke Mood Detection — Live Demo", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN_CSV = os.path.join(BASE_DIR, "window_features.csv")
MODEL_PKL = os.path.join(BASE_DIR, "models", "best_model.pkl")

SYSTEM_PROMPT = (
    "Given the user's current keystroke statistics, personal baseline deviations, "
    "recent trends, and ML prediction, provide a concise feature-grounded interpretation. "
    "Rules: use ONLY supplied numbers; do not invent user info; do not claim certainty; "
    "treat ML prediction as input not truth; state uncertainty; avoid medical diagnosis; "
    "never mention typed content."
)

CORE = ["typing_speed", "mean_dwell", "mean_flight", "mean_pause", "pause_count", "correction_rate"]


@st.cache_resource
def load_assets():
    bundle = pickle.load(open(MODEL_PKL, "rb"))
    df = pd.read_csv(WIN_CSV, parse_dates=["window_start"])
    return bundle["model"], bundle["features"], bundle["labels"], df


def grounded_explanation(d):
    c, z, t, ml = d["current"], d["baseline_deviation"], d["recent_trend"], d["ml_prediction"]
    feats = [
        f"typing {c['typing_speed']} keys/s ({z['typing_speed_z']:+.1f} SD vs baseline)",
        f"dwell {c['mean_dwell']} ms ({z['mean_dwell_z']:+.1f} SD)",
        f"flight {c['mean_flight']} ms ({z['mean_flight_z']:+.1f} SD)",
        f"correction rate {c['correction_rate']:.1%} ({z['correction_rate_z']:+.1f} SD)",
    ]
    trend = ("slowing" if t["typing_speed_change"] < 0 else "speeding") + f" trend ({t['typing_speed_change']:+.2f})"
    return {
        "predicted_state": ml["state"],
        "confidence": "moderate" if ml["confidence"] < 0.85 else "high",
        "explanation": (
            f"Model predicted '{ml['state']}' (p={ml['confidence']}) mainly because " + "; ".join(feats)
            + f", with a {trend}. This is a behavioural deviation from the user's own baseline, "
            "not a diagnosis, and uncertainty remains."
        ),
        "important_features": feats,
    }


def build_llm_input(row, state, conf):
    return {
        "current": {k: round(float(row[k]), 3) for k in ["typing_speed", "mean_dwell", "mean_flight", "correction_rate"]},
        "baseline_deviation": {k: round(float(row[k]), 2) for k in ["typing_speed_z", "mean_dwell_z", "mean_flight_z", "correction_rate_z"]},
        "recent_trend": {
            "typing_speed_change": round(float(row["typing_speed_change"]), 3),
            "pause_change": round(float(row["pause_count_change"]), 3),
            "correction_change": round(float(row["correction_rate_change"]), 3),
        },
        "ml_prediction": {"state": str(state), "confidence": round(float(conf), 2)},
    }


def gemini_explain(llm_input, api_key):
    from google import genai  # pip install google-genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        config={"system_instruction": SYSTEM_PROMPT},
        contents=json.dumps(llm_input),
    )
    return resp.text


# ---------- Load ----------
try:
    model, FEATS, LABELS, wdf = load_assets()
except Exception as e:
    st.error(f"Could not load assets: {e}\nExpected {WIN_CSV} and {MODEL_PKL}. Run the notebook first.")
    st.stop()

st.title("⌨️ LLM-Assisted Mood Detection — Live Demo")
st.info("🔒 **Privacy:** Typed content: **NOT COLLECTED**. Only timing/behavioural metadata is used. | ⚠️ Synthetic benchmark — not real-world accuracy, no medical claims.")

# ---------- Sidebar ----------
st.sidebar.header("Demo controls")
mode = st.sidebar.radio("Mode", ["Real window from dataset", "Simulate typing (sliders)"], index=0)
users = sorted(wdf["user_id"].unique())
sel_user = st.sidebar.selectbox("Baseline user", users, index=0)
gemini_key = st.sidebar.text_input("Gemini API key (optional)", type="password", help="Paste to enable live Gemini explanation. Else offline explainer is used.")

if mode == "Real window from dataset":
    uw = wdf[wdf["user_id"] == sel_user].sort_values("window_start").reset_index(drop=True)
    idx = st.sidebar.slider(f"Window # (0–{len(uw)-1})", 0, len(uw) - 1, min(3, len(uw) - 1))
    row = uw.iloc[idx].copy()
else:
    base_row = wdf[wdf["user_id"] == sel_user].median(numeric_only=True)
    st.sidebar.markdown("**Simulated behaviour** (deviations computed vs this user's baseline)")
    ts = st.sidebar.slider("typing_speed (keys/s)", 1.0, 8.0, float(round(base_row["typing_speed"], 2)))
    dw = st.sidebar.slider("mean_dwell (ms)", 50.0, 200.0, float(round(base_row["mean_dwell"], 1)))
    fl = st.sidebar.slider("mean_flight (ms)", 50.0, 500.0, float(round(base_row["mean_flight"], 1)))
    cr = st.sidebar.slider("correction_rate", 0.0, 0.30, float(round(base_row["correction_rate"], 3)))
    pc = st.sidebar.slider("pause_count", 0, 10, int(base_row["pause_count"]))
    row = wdf[wdf["user_id"] == sel_user].sort_values("window_start").iloc[len(wdf[wdf["user_id"] == sel_user]) // 2].copy()
    row.update({"typing_speed": ts, "mean_dwell": dw, "mean_flight": fl, "correction_rate": cr, "pause_count": pc,
                "typing_speed_change": 0.0, "mean_dwell_change": 0.0, "mean_flight_change": 0.0,
                "pause_count_change": 0.0, "correction_rate_change": 0.0})
    # recompute z-scores vs user baseline
    umean = wdf[wdf["user_id"] == sel_user][CORE].mean()
    ustd = wdf[wdf["user_id"] == sel_user][CORE].std().replace(0, 1)
    for c in CORE:
        row[f"{c}_z"] = (row[c] - umean[c]) / ustd[c]

# ---------- Predict ----------
X = pd.DataFrame([row])[FEATS]
proba = model.predict_proba(X)[0]
pred_idx = int(proba.argmax())
pred_state, pred_conf = LABELS[pred_idx], float(proba[pred_idx])
llm_in = build_llm_input(row, pred_state, pred_conf)
offline = grounded_explanation(llm_in)

# ---------- Current state ----------
c1, c2, c3 = st.columns(3)
c1.metric("Predicted State", pred_state.upper(), f"{pred_conf:.0%} confidence")
c2.metric("True label (synthetic)", str(row.get("mood_state", "n/a")))
c3.metric("User", sel_user)

# ---------- Features + baseline ----------
st.subheader("Current features")
f1, f2, f3, f4 = st.columns(4)
f1.metric("Typing speed", f"{row['typing_speed']:.1f} keys/s", f"{row['typing_speed_z']:+.1f} SD")
f2.metric("Mean dwell", f"{row['mean_dwell']:.0f} ms", f"{row['mean_dwell_z']:+.1f} SD")
f3.metric("Mean flight", f"{row['mean_flight']:.0f} ms", f"{row['mean_flight_z']:+.1f} SD")
f4.metric("Correction rate", f"{row['correction_rate']:.1%}", f"{row['correction_rate_z']:+.1f} SD")

with st.expander("All model features (JSON sent to LLM)"):
    st.json(llm_in)

# ---------- Temporal trend ----------
st.subheader("Temporal trend (recent windows for this user)")
hist = wdf[wdf["user_id"] == sel_user].sort_values("window_start").tail(8)
st.line_chart(hist.set_index("window_start")[["typing_speed", "correction_rate"]])

# ---------- Explanation ----------
st.subheader("Explanation (feature-grounded, no typed content)")
st.write(offline["explanation"])
st.caption("Important features: " + " • ".join(offline["important_features"]))

if gemini_key:
    if st.button("✨ Generate live Gemini explanation"):
        with st.spinner("Asking Gemini..."):
            try:
                st.markdown(gemini_explain(llm_in, gemini_key))
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
else:
    st.caption("Add a Gemini API key in the sidebar to show a live LLM explanation alongside the offline one.")

# ---------- Model transparency ----------
with st.expander("Model transparency (feature importance)"):
    try:
        imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False).head(12)
        st.bar_chart(imp)
    except Exception as e:
        st.write(f"Unavailable: {e}")

st.divider()
st.caption("Viva line: controlled synthetic benchmark to validate the full pipeline (cleaning → EDA → windowing → baseline → temporal → ML → LLM). Not a claim of real-world emotion accuracy.")
