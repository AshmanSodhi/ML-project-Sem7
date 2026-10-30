"""Live keystroke mood detection — type, pause 3s, get emotion.
Run: streamlit run app/live_typing.py

Privacy: raw typed TEXT never leaves the browser. Only timing stats
(dwell/flight summaries, correction count) are sent to Python.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Keystroke Mood Detection", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN_CSV = os.path.join(BASE_DIR, "window_features.csv")
MODEL_PKL = os.path.join(BASE_DIR, "models", "best_model.pkl")

ZBASE = ["typing_speed", "mean_dwell", "mean_flight", "mean_pause", "pause_count", "correction_rate"]
TCHG_MAP = [("typing_speed", "typing_speed_change"), ("mean_dwell", "mean_dwell_change"),
            ("mean_flight", "mean_flight_change"), ("pause_count", "pause_count_change"),
            ("correction_rate", "correction_rate_change")]
EMOJI = {"calm": "😌", "positive": "😊", "stressed": "😟", "low": "😔"}

SYSTEM_PROMPT = (
    "Given the user's current keystroke statistics, personal baseline deviations, "
    "recent trends, and ML prediction, provide a concise feature-grounded interpretation. "
    "Rules: use ONLY supplied numbers; do not invent user info; do not claim certainty; "
    "treat ML prediction as input not truth; state uncertainty; avoid medical diagnosis; "
    "never mention typed content."
)


@st.cache_resource
def load_assets():
    import pickle
    bundle = pickle.load(open(MODEL_PKL, "rb"))
    df = pd.read_csv(WIN_CSV, parse_dates=["window_start"])
    return bundle["model"], bundle["features"], bundle["labels"], df


def payload_to_row(payload, user_id, wdf, history):
    """Convert browser timing stats -> one model-ready feature row."""
    dw = np.array(payload["dwells"], dtype=float)
    fl = np.array(payload.get("flights", []), dtype=float)
    if fl.size == 0:
        fl = np.array([0.0])
    n = int(payload["n_keys"])
    dur = max(float(payload["duration_sec"]), 0.5)
    row = {
        "typing_speed": n / dur,
        "mean_dwell": float(dw.mean()), "std_dwell": float(dw.std()) if len(dw) > 1 else 0.0,
        "mean_flight": float(fl.mean()), "std_flight": float(fl.std()) if len(fl) > 1 else 0.0,
        "mean_pause": float(fl[fl > 500].mean()) if (fl > 500).any() else 0.0,
        "pause_count": int((fl > 500).sum()), "long_pause_count": int((fl > 1000).sum()),
        "correction_rate": float(payload["corrections"]) / max(n, 1), "n_keys": n,
    }
    # z-scores vs this user's baseline
    u = wdf[wdf["user_id"] == user_id][ZBASE]
    umean, ustd = u.mean(), u.std().replace(0, 1)
    for c in ZBASE:
        row[f"{c}_z"] = float((row[c] - umean[c]) / ustd[c])
    # temporal change vs mean of last <=3 live windows (0 if none yet)
    for feat, chg in TCHG_MAP:
        if history:
            prev = np.mean([h["row"][feat] for h in history[-3:]])
            row[chg] = float(row[feat] - prev)
        else:
            row[chg] = 0.0
    return row


def grounded_explanation(llm_in):
    c, z, t, ml = llm_in["current"], llm_in["baseline_deviation"], llm_in["recent_trend"], llm_in["ml_prediction"]
    feats = [
        f"typing {c['typing_speed']} keys/s ({z['typing_speed_z']:+.1f} SD vs baseline)",
        f"dwell {c['mean_dwell']} ms ({z['mean_dwell_z']:+.1f} SD)",
        f"flight {c['mean_flight']} ms ({z['mean_flight_z']:+.1f} SD)",
        f"correction rate {c['correction_rate']:.1%} ({z['correction_rate_z']:+.1f} SD)",
    ]
    trend = ("slowing" if t["typing_speed_change"] < 0 else "speeding") + f" trend ({t['typing_speed_change']:+.2f})"
    conf = "moderate" if ml["confidence"] < 0.85 else "high"
    return {
        "explanation": (f"Model predicted '{ml['state']}' (p={ml['confidence']}) mainly because "
                        + "; ".join(feats) + f", with a {trend}. Behavioural deviation from "
                        "the user's own baseline — not a diagnosis, uncertainty remains."),
        "confidence_word": conf, "features": feats,
    }


# ---------- Load ----------
try:
    model, FEATS, LABELS, wdf = load_assets()
except Exception as e:
    st.error(f"Could not load assets: {e}")
    st.stop()

st.title("⌨️ Live Mood Detection — type, pause, see emotion")
st.info("🔒 **Privacy:** what you type **never leaves this browser tab** — only key-timing stats are used. "
        "⚠️ Synthetic-data model: classroom demo, not a medical device.")

# ---------- Sidebar ----------
st.sidebar.header("Live demo controls")
sel_user = st.sidebar.selectbox("Compare against user baseline", sorted(wdf["user_id"].unique()), index=0,
                                help="Deviations are computed vs THIS synthetic user's normal typing.")
gap_s = st.sidebar.slider("Silence gap to trigger prediction (s)", 1, 6, 3)
gemini_key = st.sidebar.text_input("Gemini API key (optional)", type="password")
if st.sidebar.button("Reset session"):
    st.session_state.live_windows = []
    st.rerun()

listener = components.declare_component(
    "keystroke_listener",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "keystroke_component"),
)

st.markdown("### 1️⃣ Type here (click inside the box first — paste won't count, type naturally)")
data = listener(gap_ms=gap_s * 1000, default=None, key="ks-live", height=230)

if "live_windows" not in st.session_state:
    st.session_state.live_windows = []

# ---------- New window arrived -> predict ----------
if data is not None and (not st.session_state.live_windows or
                         st.session_state.live_windows[-1]["win_id"] != data.get("win_id")):
    row = payload_to_row(data, sel_user, wdf, st.session_state.live_windows)
    proba = model.predict_proba(pd.DataFrame([row])[FEATS])[0]
    pred, conf = LABELS[int(proba.argmax())], float(proba.max())
    llm_in = {
        "current": {k: round(float(row[k]), 3) for k in ["typing_speed", "mean_dwell", "mean_flight", "correction_rate"]},
        "baseline_deviation": {k: round(float(row[k]), 2)
                               for k in ["typing_speed_z", "mean_dwell_z", "mean_flight_z", "correction_rate_z"]},
        "recent_trend": {"typing_speed_change": round(float(row["typing_speed_change"]), 3),
                         "pause_change": round(float(row["pause_count_change"]), 3),
                         "correction_change": round(float(row["correction_rate_change"]), 3)},
        "ml_prediction": {"state": pred, "confidence": round(conf, 2)},
    }
    st.session_state.live_windows.append(
        {"win_id": data.get("win_id"), "row": row, "pred": pred, "conf": conf,
         "proba": proba, "llm_in": llm_in, "expl": grounded_explanation(llm_in)})
    st.rerun()

# ---------- Display ----------
wins = st.session_state.live_windows
st.markdown("### 2️⃣ Emotion (appears after you stop typing)")
if not wins:
    st.markdown("## ⌨️ _Waiting… type above, then pause._")
else:
    last = wins[-1]
    st.markdown(f"# {EMOJI.get(last['pred'], '')} {last['pred'].upper()} — {last['conf']:.0%}")
    st.caption(f"Window: {last['row']['n_keys']} keys in "
               f"{last['row']['n_keys']/max(last['row']['typing_speed'], 1e-3):.1f}s "
               f"| vs baseline {sel_user} | ground-truth unknown (live data)")
    st.progress(min(last["conf"], 1.0))
    with st.columns(2)[0]:
        st.write("**All class probabilities**")
        st.bar_chart(pd.Series(last["proba"], index=LABELS))

    c1, c2, c3, c4 = st.columns(4)
    r = last["row"]
    c1.metric("Typing speed", f"{r['typing_speed']:.1f} keys/s", f"{r['typing_speed_z']:+.1f} SD")
    c2.metric("Mean dwell", f"{r['mean_dwell']:.0f} ms", f"{r['mean_dwell_z']:+.1f} SD")
    c3.metric("Mean flight", f"{r['mean_flight']:.0f} ms", f"{r['mean_flight_z']:+.1f} SD")
    c4.metric("Corrections", f"{r['correction_rate']:.1%}", f"{r['correction_rate_z']:+.1f} SD")

    st.markdown("### 3️⃣ Why? (feature-grounded)")
    st.write(last["expl"]["explanation"])
    st.caption("Key signals: " + " • ".join(last["expl"]["features"]))

    if gemini_key:
        if st.button("✨ Explain with Gemini"):
            with st.spinner("Asking Gemini…"):
                try:
                    from google import genai  # pip install google-genai
                    client = genai.Client(api_key=gemini_key)
                    resp = client.models.generate_content(
                        model="gemini-2.0-flash",
                        config={"system_instruction": SYSTEM_PROMPT},
                        contents=json.dumps(last["llm_in"]))
                    st.markdown(resp.text)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")

    if len(wins) > 1:
        st.markdown("### Session history")
        hist = pd.DataFrame([{"window": i + 1, "pred": w["pred"], "conf": round(w["conf"], 2),
                              "speed": round(w["row"]["typing_speed"], 2)} for i, w in enumerate(wins)])
        st.dataframe(hist, use_container_width=True)
        st.line_chart(hist.set_index("window")[["conf"]])
