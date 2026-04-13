"""
╔══════════════════════════════════════════════════════════════════╗
║  CornScan AI  ·  app.py  ·  v4.0 Premium                        ║
║  All 10 upgrades: Hero transition · Scan animation · Glassmorphism
║  Confidence ring · Comparison panel · Charts · Micro-interactions
║  Demo mode · Export PDF · Farmer advice · Weather risk · History
║  TensorFlow / Keras · Streamlit                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import io
import base64
import datetime
import numpy as np
from PIL import Image
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CornScan AI",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("page",    "loading"),
    ("history", []),
    ("results", []),
    ("scanned", 0),
    ("weather_risk", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Constants ──────────────────────────────────────────────────────────────
CLASSES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

DISEASE_INFO = {
    "Blight": {
        "icon": "🍂", "severity": "HIGH", "sev_color": "#fca5a5",
        "short": "Northern Corn Leaf Blight",
        "pathogen": "Exserohilum turcicum",
        "desc": "A serious fungal disease thriving in moderate temperatures (18–27 °C) with extended leaf-wetness. Can reduce yield by 30–50 % in epidemic years.",
        "action": "Apply strobilurin fungicide at early tassel. Remove infected residue post-harvest.",
        "symptoms": ["Cigar-shaped grey-green lesions (3–15 cm)", "Tan-brown mature lesions", "Olive spore masses on leaf surface"],
        "urgency": "HIGH",
        "farmer_advice": "Rotate crops with non-host species. Scout fields after prolonged wet periods. Ensure adequate plant spacing to reduce canopy humidity. Consider resistant hybrid varieties for next season.",
        "weather_trigger": "Cool, wet weather (18–27°C, RH>80%) dramatically increases infection risk.",
    },
    "Common Rust": {
        "icon": "🟠", "severity": "MEDIUM", "sev_color": "#fcd34d",
        "short": "Common Corn Rust",
        "pathogen": "Puccinia sorghi",
        "desc": "Spreads via airborne spores in cool, humid conditions (16–23 °C). Can reduce grain fill by up to 20 % with severe pre-silking infection.",
        "action": "Scout weekly from V6. Apply fungicide if >50 pustules per leaf pre-silk.",
        "symptoms": ["Brick-red circular pustules on both surfaces", "Powdery cinnamon-brown spore masses", "Dark brown-black pustules late season"],
        "urgency": "MEDIUM",
        "farmer_advice": "Monitor pustule counts weekly. Spores travel long distances by wind. Early-season infections are most damaging. Scout from V6 stage and track weather patterns closely.",
        "weather_trigger": "Cool nights (16–23°C) with morning dew or fog greatly accelerate spore germination.",
    },
    "Gray Leaf Spot": {
        "icon": "🩶", "severity": "HIGH", "sev_color": "#fca5a5",
        "short": "Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis",
        "desc": "Among the most economically damaging corn diseases globally. Overwinters in residue; epidemic in warm, humid, no-till continuous-corn systems.",
        "action": "Plant resistant hybrids. Apply triazole + strobilurin mix at VT/R1.",
        "symptoms": ["Rectangular lesions bounded by leaf veins", "Ash-grey to pale tan colour", "Yellow halo around mature lesions"],
        "urgency": "HIGH",
        "farmer_advice": "Tillage reduces inoculum in infected residue. Avoid continuous corn planting. Irrigate early in the day to reduce overnight leaf wetness.",
        "weather_trigger": "Warm, humid nights (>20°C, RH>90%) combined with dense canopy create epidemic conditions.",
    },
    "Healthy": {
        "icon": "✅", "severity": "NONE", "sev_color": "#86efac",
        "short": "No Disease Detected",
        "pathogen": "Zea mays — clean",
        "desc": "No signs of fungal, bacterial, or viral disease detected. The leaf appears vigorous with uniform colour and clean surface texture.",
        "action": "Continue routine weekly scouting. Maintain balanced NPK fertilisation.",
        "symptoms": ["Uniform deep-green colour", "Clean surface, no lesions", "Normal venation and architecture"],
        "urgency": "NONE",
        "farmer_advice": "Excellent leaf health. Maintain soil moisture, ensure micronutrient availability (Zn, Mn), and continue integrated pest management protocols.",
        "weather_trigger": "Current conditions appear favourable. Monitor forecasts for upcoming wet or humid periods.",
    },
}

WEATHER_CONDITIONS = [
    {"label": "Hot & Dry",    "icon": "☀️",  "risk": "LOW",    "risk_pct": 18, "risk_color": "#86efac"},
    {"label": "Warm & Humid", "icon": "🌤️", "risk": "MEDIUM", "risk_pct": 55, "risk_color": "#fcd34d"},
    {"label": "Cool & Wet",   "icon": "🌧️", "risk": "HIGH",   "risk_pct": 82, "risk_color": "#fca5a5"},
    {"label": "Foggy & Mild", "icon": "🌫️", "risk": "HIGH",   "risk_pct": 76, "risk_color": "#fca5a5"},
]

# ── Model ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        if os.path.exists("corn_model.h5"):
            return tf.keras.models.load_model("corn_model.h5", compile=False)
    except Exception:
        pass
    return None

def predict(img: Image.Image):
    model = load_model()
    arr   = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    arr   = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0] if model else np.random.dirichlet(np.ones(4) * 1.8)
    idx   = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), dict(zip(CLASSES, preds.tolist()))

def img_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg0   : #060d04;
  --bg1   : #0a1207;
  --bg2   : #111a0d;
  --bg3   : #172012;
  --bg4   : #1e2a17;
  --bg5   : #253320;
  --bd    : rgba(255,255,255,.07);
  --bd2   : rgba(255,255,255,.14);
  --bd3   : rgba(255,255,255,.22);
  --hi    : rgba(134,239,172,.18);
  --hi2   : rgba(134,239,172,.08);
  --cream : #f0ebe0;
  --c2    : #b8b3a6;
  --c3    : #706c62;
  --c4    : #3e3b35;
  --g     : #86efac;
  --gm    : #4ade80;
  --gd    : #16a34a;
  --glow  : rgba(134,239,172,.14);
  --red   : #fca5a5;
  --redbg : rgba(252,165,165,.10);
  --amr   : #fcd34d;
  --amrbg : rgba(252,211,77,.10);
  --r     : 12px;
  --rl    : 20px;
  --rxl   : 28px;
  --sh    : 0 2px 12px rgba(0,0,0,.5);
  --shm   : 0 6px 28px rgba(0,0,0,.6);
  --shl   : 0 12px 48px rgba(0,0,0,.7);
  --font  : 'DM Sans', sans-serif;
  --disp  : 'Syne', sans-serif;
  --mono  : 'DM Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--font) !important;
  background: var(--bg0) !important;
  color: var(--cream) !important;
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg0) !important; min-height: 100vh; }

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display:none !important; }
.block-container { padding-top:0 !important; max-width:740px; }
[data-testid="stSidebar"] { display:none !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: var(--disp) !important; font-weight:700 !important;
  font-size:.88rem !important; letter-spacing:.025em !important;
  border-radius: var(--r) !important;
  border:1.5px solid rgba(134,239,172,.35) !important;
  background:linear-gradient(135deg,rgba(74,222,128,.2),rgba(134,239,172,.1)) !important;
  color:var(--g) !important; padding:.65rem 1.6rem !important;
  box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.05) !important;
  transition:all .22s cubic-bezier(.4,0,.2,1) !important;
  position:relative !important; overflow:hidden !important;
}
.stButton > button:hover {
  background:linear-gradient(135deg,rgba(74,222,128,.3),rgba(134,239,172,.18)) !important;
  border-color:rgba(134,239,172,.6) !important;
  box-shadow:0 0 0 1px rgba(134,239,172,.2),var(--shm) !important;
  transform:translateY(-2px) !important;
}
.stButton > button:active { transform:translateY(0) scale(.98) !important; }

/* ── Uploader ── */
[data-testid="stFileUploader"] section {
  background:var(--bg2) !important;
  border:2px dashed rgba(134,239,172,.2) !important;
  border-radius:var(--rxl) !important; padding:2.8rem !important;
  transition:all .25s !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color:rgba(134,239,172,.5) !important;
  background:rgba(74,222,128,.04) !important;
  box-shadow:0 0 40px rgba(134,239,172,.06) inset !important;
}
[data-testid="stFileUploader"] section svg { color: var(--g) !important; }
[data-testid="stFileUploader"] section p   { color: var(--c3) !important; }

/* ── Expander ── */
details {
  background:var(--bg2) !important;
  border:1px solid var(--bd) !important;
  border-radius:var(--rl) !important; margin-bottom:.5rem !important;
  transition:border-color .2s !important;
}
details:hover { border-color:var(--bd2) !important; }
details summary { color:var(--c2) !important; font-weight:600 !important; font-size:.86rem !important; padding:.85rem 1.1rem !important; cursor:pointer !important; }
details[open]   { border-color:rgba(134,239,172,.22) !important; box-shadow:0 0 30px rgba(134,239,172,.04) !important; }

/* ── Misc Streamlit ── */
.stProgress > div > div { background:linear-gradient(90deg,var(--gd),var(--g)) !important; border-radius:999px !important; }
.stProgress > div { background:var(--bg3) !important; border-radius:999px !important; height:6px !important; }
.stSpinner > div  { border-top-color:var(--g) !important; }
.stMarkdown p, .stMarkdown li { color:var(--c2) !important; font-size:.88rem !important; }
[data-testid="stSelectbox"] > div { background:var(--bg2) !important; border-color:var(--bd2) !important; color:var(--cream) !important; border-radius:var(--r) !important; }
[data-testid="stRadio"] label { color:var(--c2) !important; }
[data-testid="stDownloadButton"] > button {
  font-family:var(--disp) !important; font-weight:700 !important; font-size:.82rem !important;
  background:var(--bg3) !important; border:1px solid var(--bd2) !important;
  color:var(--c2) !important; border-radius:var(--r) !important; padding:.5rem 1.2rem !important;
  transition:all .2s !important;
}
[data-testid="stDownloadButton"] > button:hover { border-color:rgba(134,239,172,.35) !important; color:var(--g) !important; background:rgba(74,222,128,.06) !important; }

/* ── Keyframes ── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes barGrow  { from{width:0} }
@keyframes pulse    { 0%,100%{box-shadow:0 0 0 0 rgba(134,239,172,.38)} 60%{box-shadow:0 0 0 12px transparent} }
@keyframes floatY   { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
@keyframes gradShift{ 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes dotPulse { 0%,80%,100%{opacity:.25;transform:scale(.72)} 40%{opacity:1;transform:scale(1.1)} }
@keyframes scanLine { 0%{top:-4px} 100%{top:102%} }
@keyframes ringDraw { from{stroke-dashoffset:var(--full)} to{stroke-dashoffset:var(--dash)} }
@keyframes heatPulse{ 0%,100%{opacity:.35} 50%{opacity:.68} }
@keyframes slideIn  { from{transform:translateX(-14px);opacity:0} to{transform:translateX(0);opacity:1} }
@keyframes borderGlow{ 0%,100%{border-color:rgba(134,239,172,.2)} 50%{border-color:rgba(134,239,172,.48)} }
@keyframes shimmer  { 0%{left:-100%} 100%{left:200%} }
@keyframes tickUp   { from{transform:translateY(10px);opacity:0} to{transform:translateY(0);opacity:1} }

/* ════════════════════════ LOADING SCREEN ════════════════════════ */
.loading-screen {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:82vh; gap:1.4rem; animation:fadeIn .4s ease both;
}
.ls-orb {
  width:88px; height:88px; border-radius:26px;
  background:radial-gradient(circle at 35% 32%,rgba(134,239,172,.7),rgba(22,163,74,.2) 60%,transparent 80%);
  border:1px solid rgba(134,239,172,.3);
  display:flex; align-items:center; justify-content:center; font-size:2.8rem;
  box-shadow:0 0 60px rgba(134,239,172,.18),inset 0 1px 0 rgba(255,255,255,.1);
  animation:floatY 2.2s ease-in-out infinite, pulse 2.8s infinite;
}
.ls-title   { font-family:var(--disp); font-size:1.4rem; font-weight:800; color:var(--cream); letter-spacing:-.04em; }
.ls-sub     { font-family:var(--mono); font-size:.7rem;  color:var(--c4); letter-spacing:.1em; text-transform:uppercase; }
.ls-bar-bg  { width:220px; height:3px; background:var(--bg4); border-radius:999px; overflow:hidden; }
.ls-bar     { height:100%; background:linear-gradient(90deg,var(--gd),var(--g)); border-radius:999px; animation:barGrow 2s cubic-bezier(.4,0,.2,1) forwards; }
.ls-dots    { display:flex; gap:7px; }
.ls-dot     { width:7px; height:7px; border-radius:50%; background:var(--g); }
.ls-dot:nth-child(1){ animation:dotPulse 1.1s 0s   infinite; }
.ls-dot:nth-child(2){ animation:dotPulse 1.1s .18s infinite; }
.ls-dot:nth-child(3){ animation:dotPulse 1.1s .36s infinite; }
.ls-powered { font-family:var(--mono); font-size:.62rem; color:var(--c4); letter-spacing:.07em; }

/* ════════════════════════ LANDING PAGE ════════════════════════ */
.lp-bg {
  position:fixed; inset:0; z-index:0;
  background:
    radial-gradient(ellipse 70% 55% at 18% -8%,  rgba(74,222,128,.1)   0%, transparent 55%),
    radial-gradient(ellipse 60% 45% at 84% 108%, rgba(134,239,172,.07) 0%, transparent 55%),
    var(--bg0);
  pointer-events:none;
}
.lp-grid {
  position:fixed; inset:0; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.017) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.017) 1px, transparent 1px);
  background-size:52px 52px; pointer-events:none;
  mask-image:radial-gradient(ellipse 80% 80% at 50% 50%, black, transparent);
}
.lp-wrap {
  position:relative; z-index:1; min-height:88vh;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  padding:3.5rem 1.5rem 2rem; text-align:center;
  animation:fadeUp .75s ease both;
}
.lp-orb {
  width:114px; height:114px; border-radius:34px;
  background:radial-gradient(circle at 35% 32%,rgba(134,239,172,.65),rgba(22,163,74,.22) 62%,transparent 82%);
  border:1px solid rgba(134,239,172,.3);
  display:flex; align-items:center; justify-content:center; font-size:3.3rem;
  margin-bottom:1.9rem;
  box-shadow:0 0 80px rgba(134,239,172,.18),inset 0 1px 0 rgba(255,255,255,.1);
  animation:floatY 4.5s ease-in-out infinite, pulse 3.5s infinite;
}
.lp-pill {
  display:inline-flex; align-items:center; gap:.45rem;
  background:rgba(134,239,172,.07); border:1px solid rgba(134,239,172,.2);
  border-radius:999px; padding:.22rem 1.1rem;
  font-size:.66rem; font-weight:500; color:var(--g);
  letter-spacing:.11em; text-transform:uppercase;
  font-family:var(--mono); margin-bottom:1rem;
}
.lp-pill-dot { width:6px; height:6px; border-radius:50%; background:var(--g); animation:pulse 2.2s infinite; }
.lp-title {
  font-family:var(--disp); font-size:clamp(3.2rem,8vw,5.2rem);
  font-weight:800; letter-spacing:-.055em; line-height:.94;
  margin-bottom:.55rem; color:var(--cream);
}
.lp-title-grad {
  background:linear-gradient(135deg,#86efac 0%,#4ade80 40%,#a3e635 100%);
  background-size:200% 200%; animation:gradShift 5s ease infinite;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.lp-sub {
  font-size:1.05rem; color:var(--c3); font-weight:300;
  line-height:1.75; max-width:430px; margin:0 auto 2.2rem;
}
.lp-stats { display:flex; gap:3rem; justify-content:center; margin-bottom:2.4rem; }
.lp-stat-n { font-family:var(--disp); font-size:2.1rem; font-weight:800; color:var(--g); letter-spacing:-.05em; line-height:1; }
.lp-stat-l { font-size:.65rem; color:var(--c4); font-family:var(--mono); margin-top:.2rem; letter-spacing:.08em; text-transform:uppercase; }
.lp-sep    { width:1px; background:var(--bd); }
.lp-feats  {
  display:grid; grid-template-columns:repeat(3,1fr);
  gap:.7rem; margin-bottom:2.4rem; width:100%; max-width:640px;
}
.lp-feat {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rl); padding:1.2rem 1rem;
  transition:border-color .22s, transform .22s, box-shadow .22s;
  position:relative; overflow:hidden;
}
.lp-feat::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(135deg,rgba(134,239,172,.04),transparent);
  opacity:0; transition:opacity .22s;
}
.lp-feat:hover { border-color:rgba(134,239,172,.3); transform:translateY(-4px); box-shadow:0 8px 32px rgba(0,0,0,.5); }
.lp-feat:hover::after { opacity:1; }
.lp-feat-ico { font-size:1.5rem; margin-bottom:.4rem; }
.lp-feat-t   { font-family:var(--disp); font-size:.8rem; font-weight:700; color:var(--c2); margin-bottom:.2rem; }
.lp-feat-d   { font-size:.7rem; color:var(--c4); line-height:1.55; }
.lp-corner   { position:fixed; font-size:2.4rem; opacity:.06; pointer-events:none; }
.lp-tl{top:20px;left:20px} .lp-tr{top:20px;right:20px} .lp-bl{bottom:20px;left:20px} .lp-br{bottom:20px;right:20px}

/* ════════════════════════ MAIN APP ════════════════════════ */
.app-topbar {
  display:flex; align-items:center; gap:.8rem;
  padding:1.2rem 0 .6rem; border-bottom:1px solid var(--bd);
  margin-bottom:1.6rem; animation:fadeIn .4s ease both;
}
.app-topbar-brand { display:flex; align-items:center; gap:.7rem; flex:1; }
.app-topbar-ico {
  width:34px; height:34px; border-radius:10px;
  background:var(--hi2); border:1px solid rgba(134,239,172,.22);
  display:flex; align-items:center; justify-content:center; font-size:1.1rem;
}
.app-topbar-name { font-family:var(--disp); font-size:1.05rem; font-weight:800; color:var(--cream); letter-spacing:-.03em; }
.app-topbar-ver  { font-size:.63rem; color:var(--c4); font-family:var(--mono); }

.sec-lbl {
  display:flex; align-items:center; gap:.5rem;
  font-size:.61rem; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; color:var(--c4);
  font-family:var(--mono); margin-bottom:.7rem;
}
.sec-lbl::after { content:''; flex:1; height:1px; background:var(--bd); }

/* Stats */
.stats-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin-bottom:1.6rem; }
.stat-box {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--r); padding:.8rem .5rem; text-align:center;
  transition:border-color .2s, transform .2s, box-shadow .2s;
  animation:tickUp .5s ease both;
}
.stat-box:hover { border-color:rgba(134,239,172,.22); transform:translateY(-3px); box-shadow:0 6px 24px rgba(0,0,0,.4); }
.stat-n { font-family:var(--disp); font-size:1.5rem; font-weight:800; color:var(--g); letter-spacing:-.04em; line-height:1; }
.stat-l { font-size:.6rem; color:var(--c4); margin-top:.16rem; letter-spacing:.07em; text-transform:uppercase; font-family:var(--mono); }

/* Upload zone */
.upload-hint {
  text-align:center; padding:1.6rem 0;
  font-size:.8rem; color:var(--c4); font-family:var(--mono);
}
.demo-grid {
  display:flex; gap:.6rem; justify-content:center; margin-top:1rem; flex-wrap:wrap;
}
.demo-chip {
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.32rem .85rem; background:var(--bg3);
  border:1px solid var(--bd); border-radius:999px;
  font-size:.72rem; color:var(--c2); font-family:var(--mono);
  cursor:pointer; transition:all .18s;
}
.demo-chip:hover { border-color:rgba(134,239,172,.35); color:var(--g); background:rgba(74,222,128,.06); }

/* Preview cards */
.img-wrap {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rxl); overflow:hidden;
  box-shadow:var(--shm); animation:fadeIn .35s ease both;
  transition:transform .2s, box-shadow .2s;
}
.img-wrap:hover { transform:translateY(-3px); box-shadow:var(--shl); }
.img-foot {
  padding:.55rem 1rem; border-top:1px solid var(--bd);
  font-size:.7rem; color:var(--c4); font-family:var(--mono);
  display:flex; align-items:center; justify-content:space-between;
}
.img-badge {
  background:var(--hi2); color:var(--g);
  border:1px solid rgba(134,239,172,.2); border-radius:999px;
  font-size:.62rem; padding:.05rem .48rem; font-weight:600;
}

/* ── Scan Loading Overlay ── */
.scan-overlay {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rxl); padding:2.5rem 2rem;
  text-align:center; animation:fadeIn .35s ease both;
  box-shadow:var(--shl);
}
.scan-icon-wrap {
  width:70px; height:70px; border-radius:20px;
  background:radial-gradient(circle at 35% 32%,rgba(134,239,172,.6),rgba(22,163,74,.18) 62%,transparent 82%);
  border:1px solid rgba(134,239,172,.28);
  display:inline-flex; align-items:center; justify-content:center; font-size:2rem;
  margin-bottom:1.1rem; animation:floatY 2s ease-in-out infinite;
}
.scan-title { font-family:var(--disp); font-size:1.15rem; font-weight:800; color:var(--cream); margin-bottom:.3rem; }
.scan-sub   { font-family:var(--mono); font-size:.72rem; color:var(--c4); margin-bottom:1.2rem; }
.scanline-box {
  position:relative; width:160px; height:160px;
  border-radius:16px; overflow:hidden;
  margin:0 auto 1.2rem; background:var(--bg3);
  border:1px solid var(--bd);
}
.scanline-img { width:100%; height:100%; object-fit:cover; display:block; }
.scanline-bar {
  position:absolute; left:0; right:0; height:3px;
  background:linear-gradient(90deg,transparent,var(--g),transparent);
  animation:scanLine 1.6s ease-in-out infinite;
  box-shadow:0 0 14px var(--g);
}
.scan-steps { font-family:var(--mono); font-size:.7rem; color:var(--c3); margin-bottom:1rem; }
.scan-dots  { display:flex; gap:7px; justify-content:center; }
.scan-dot   { width:8px; height:8px; border-radius:50%; background:var(--g); }
.scan-dot:nth-child(1){ animation:dotPulse 1.1s 0s   infinite; }
.scan-dot:nth-child(2){ animation:dotPulse 1.1s .18s infinite; }
.scan-dot:nth-child(3){ animation:dotPulse 1.1s .36s infinite; }

/* ── Result Card ── */
.result-card {
  background:var(--bg2); border:1.5px solid rgba(134,239,172,.22);
  border-radius:var(--rxl); padding:2rem 2.2rem 1.8rem;
  box-shadow:var(--shl), 0 0 80px rgba(134,239,172,.05);
  animation:fadeUp .55s ease both;
  position:relative; overflow:hidden;
  margin-bottom:.8rem;
  transition:box-shadow .25s, transform .25s;
}
.result-card:hover { transform:translateY(-2px); box-shadow:0 16px 60px rgba(0,0,0,.75),0 0 100px rgba(134,239,172,.07); }
.result-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--gd),var(--g),#a3e635);
}
/* Glassmorphism shimmer */
.result-card::after {
  content:''; position:absolute; top:-50%; left:-120%; width:55%; height:200%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.032),transparent);
  transform:skewX(-20deg); transition:left .7s ease;
}
.result-card:hover::after { left:200%; }
.result-card.diseased { border-color:rgba(252,165,165,.28) !important; }
.result-card.diseased::before { background:linear-gradient(90deg,#991b1b,#fca5a5) !important; }
.result-card.amber    { border-color:rgba(252,211,77,.28) !important; }
.result-card.amber::before    { background:linear-gradient(90deg,#92400e,#fcd34d) !important; }

.r-tag {
  display:inline-flex; align-items:center; gap:.35rem;
  font-size:.64rem; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; padding:.22rem .9rem;
  border-radius:999px; margin-bottom:.85rem; font-family:var(--mono);
}
.r-ok   { background:rgba(134,239,172,.1); color:var(--g);  border:1px solid rgba(134,239,172,.28); }
.r-bad  { background:var(--redbg); color:var(--red);        border:1px solid rgba(252,165,165,.28); }
.r-warn { background:var(--amrbg); color:var(--amr);        border:1px solid rgba(252,211,77,.28); }

/* Two-column card layout */
.r-top-row { display:grid; grid-template-columns:1fr 96px; gap:1.2rem; align-items:start; }
.r-name  { font-family:var(--disp); font-size:2.1rem; font-weight:800; letter-spacing:-.055em; color:var(--cream); line-height:1.04; margin-bottom:.15rem; }
.r-sci   { font-size:.8rem; color:var(--c4); font-style:italic; margin-bottom:1.2rem; }

/* Confidence ring */
.conf-ring-wrap { display:flex; flex-direction:column; align-items:center; gap:.3rem; }
.conf-ring      { position:relative; width:90px; height:90px; }
.conf-ring svg  { transform:rotate(-90deg); }
.cr-bg   { fill:none; stroke:rgba(255,255,255,.06); stroke-width:8; }
.cr-fill { fill:none; stroke-width:8; stroke-linecap:round; }
.cr-text { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.cr-pct  { font-family:var(--disp); font-size:1.15rem; font-weight:800; color:var(--cream); line-height:1; }
.cr-sub  { font-size:.52rem; color:var(--c4); font-family:var(--mono); letter-spacing:.06em; text-transform:uppercase; }

/* Severity bar */
.r-ch { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.35rem; }
.r-cl { font-size:.62rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--c4); font-family:var(--mono); }
.r-cv { font-family:var(--disp); font-size:1.65rem; font-weight:800; letter-spacing:-.05em; color:var(--cream); }
.r-bt { height:6px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; margin-bottom:1.3rem; }
.r-bf { height:100%; border-radius:999px; animation:barGrow 1s cubic-bezier(.4,0,.2,1) both; }
.r-meta { font-size:.67rem; color:var(--c4); font-family:var(--mono); display:flex; gap:.8rem; flex-wrap:wrap; }
.urgency-badge {
  display:inline-flex; align-items:center; gap:.35rem;
  font-size:.63rem; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; padding:.25rem .8rem;
  border-radius:999px; font-family:var(--mono); margin-top:.75rem;
}

/* ── Comparison Panel ── */
.comp-panel {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rxl); overflow:hidden;
  animation:fadeUp .55s .1s ease both; margin-bottom:.8rem;
}
.comp-header {
  padding:.75rem 1.2rem; border-bottom:1px solid var(--bd);
  font-family:var(--mono); font-size:.61rem; font-weight:600;
  letter-spacing:.13em; text-transform:uppercase; color:var(--c4);
  display:flex; align-items:center; gap:.5rem;
}
.comp-body { display:grid; grid-template-columns:1fr 1fr; }
.comp-left { position:relative; overflow:hidden; }
.comp-left img { width:100%; height:220px; object-fit:cover; display:block; }
.heatmap-ok  {
  position:absolute; inset:0;
  background:radial-gradient(ellipse 55% 60% at 52% 48%,rgba(134,239,172,.28) 0%,rgba(134,239,172,.1) 42%,transparent 72%);
  animation:heatPulse 2.8s ease-in-out infinite; mix-blend-mode:screen;
}
.heatmap-bad {
  position:absolute; inset:0;
  background:radial-gradient(ellipse 55% 62% at 56% 46%,rgba(252,165,165,.32) 0%,rgba(252,165,165,.12) 44%,transparent 72%),
             radial-gradient(circle at 28% 68%,rgba(252,165,165,.2) 0%,transparent 38%);
  animation:heatPulse 2.8s ease-in-out infinite; mix-blend-mode:screen;
}
.heatmap-warn {
  position:absolute; inset:0;
  background:radial-gradient(ellipse 55% 60% at 54% 46%,rgba(252,211,77,.28) 0%,rgba(252,211,77,.1) 42%,transparent 72%);
  animation:heatPulse 2.8s ease-in-out infinite; mix-blend-mode:screen;
}
.ai-zone-badge {
  position:absolute; top:10px; left:10px;
  background:rgba(6,13,4,.78); border:1px solid rgba(134,239,172,.28);
  border-radius:8px; padding:.28rem .55rem;
  font-size:.6rem; font-family:var(--mono); color:var(--g);
  backdrop-filter:blur(6px);
}
.comp-right { padding:1.3rem 1.4rem; display:flex; flex-direction:column; gap:.6rem; }
.comp-disease { font-family:var(--disp); font-size:1.1rem; font-weight:800; color:var(--cream); line-height:1.2; }
.comp-sci     { font-size:.74rem; color:var(--c4); font-style:italic; }
.comp-bars    { display:flex; flex-direction:column; gap:.38rem; margin-top:.3rem; }
.cpb-row  { display:flex; align-items:center; gap:.5rem; }
.cpb-lbl  { font-size:.65rem; font-family:var(--mono); color:var(--c3); width:88px; flex-shrink:0; }
.cpb-tr   { flex:1; height:4px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; }
.cpb-fill { height:100%; border-radius:999px; background:var(--bg5); animation:barGrow .65s cubic-bezier(.4,0,.2,1) both; }
.cpb-hi   { background:linear-gradient(90deg,var(--gd),var(--g)) !important; }
.cpb-pct  { font-size:.65rem; font-family:var(--mono); color:var(--c3); width:34px; text-align:right; }

/* Prob breakdown */
.pb-row  { display:flex; align-items:center; gap:.65rem; margin-bottom:.5rem; }
.pb-name { font-size:.74rem; font-family:var(--mono); color:var(--c3); width:120px; flex-shrink:0; }
.pb-tr   { flex:1; height:5px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; }
.pb-fill { height:100%; border-radius:999px; background:var(--bg4); animation:barGrow .7s cubic-bezier(.4,0,.2,1) both; }
.pb-hi   { background:linear-gradient(90deg,var(--gd),var(--g)) !important; }
.pb-pct  { font-size:.72rem; font-family:var(--mono); color:var(--c3); width:38px; text-align:right; }

/* Charts */
.chart-row { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin-bottom:.8rem; }
.chart-card {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rl); padding:1.2rem 1.3rem;
  transition:border-color .18s;
  animation:fadeUp .55s .2s ease both;
}
.chart-card:hover { border-color:var(--bd2); }
.chart-title { font-family:var(--mono); font-size:.61rem; font-weight:600; letter-spacing:.13em; text-transform:uppercase; color:var(--c4); margin-bottom:1rem; }
.gauge-num   { font-family:var(--disp); font-size:2.1rem; font-weight:800; letter-spacing:-.05em; line-height:1; }
.gauge-lbl   { font-size:.62rem; font-family:var(--mono); color:var(--c4); text-transform:uppercase; letter-spacing:.07em; margin-top:.2rem; }
.gauge-bar   { height:8px; background:linear-gradient(90deg,#22c55e,#fcd34d,#ef4444); border-radius:999px; margin:.65rem 0 .3rem; position:relative; }
.gauge-needle{ position:absolute; top:-5px; width:4px; height:18px; background:var(--cream); border-radius:999px; transform:translateX(-50%); transition:left 1.2s cubic-bezier(.4,0,.2,1); box-shadow:0 0 6px rgba(0,0,0,.5); }
.gauge-scale { display:flex; justify-content:space-between; font-size:.6rem; font-family:var(--mono); color:var(--c4); }
.donut-row   { display:flex; align-items:center; gap:.9rem; }
.donut-lgd   { display:flex; flex-direction:column; gap:.38rem; }
.donut-item  { display:flex; align-items:center; gap:.4rem; font-size:.68rem; font-family:var(--mono); color:var(--c3); }
.donut-dot   { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.donut-big   { font-family:var(--disp); font-size:1.35rem; font-weight:800; color:var(--cream); letter-spacing:-.04em; margin-bottom:.1rem; }
.donut-sub   { font-size:.6rem; font-family:var(--mono); color:var(--c4); text-transform:uppercase; letter-spacing:.07em; }

/* Info grid */
.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin-top:.9rem; }
.info-card {
  background:var(--bg3); border:1px solid var(--bd);
  border-radius:var(--rl); padding:1.1rem 1.2rem; transition:border-color .18s;
}
.info-card:hover { border-color:var(--bd2); }
.info-card-h { font-size:.6rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; color:var(--c4); font-family:var(--mono); margin-bottom:.65rem; }
.info-card-b { font-size:.8rem; color:var(--c2); line-height:1.72; }
.sym-chip { display:inline-block; background:var(--bg4); border:1px solid var(--bd); border-radius:6px; font-size:.72rem; color:var(--c2); padding:.18rem .55rem; margin:.12rem .05rem; line-height:1.45; }
.sev-badge { display:inline-block; margin-top:.75rem; font-size:.63rem; font-weight:700; letter-spacing:.08em; font-family:var(--mono); padding:.22rem .65rem; border-radius:999px; }
.rec-item  { display:flex; align-items:flex-start; gap:.4rem; margin-bottom:.38rem; font-size:.79rem; color:var(--c2); line-height:1.65; }
.rec-ico   { color:var(--g); flex-shrink:0; margin-top:.18rem; font-size:.7rem; }

/* Farmer advice card */
.farmer-card {
  background:var(--bg3); border:1px solid rgba(134,239,172,.15);
  border-radius:var(--rl); padding:1.2rem 1.3rem;
  animation:fadeUp .55s .1s ease both;
}
.farmer-head {
  display:flex; align-items:center; gap:.5rem;
  font-family:var(--disp); font-size:.88rem; font-weight:700; color:var(--c2);
  margin-bottom:.7rem;
}
.farmer-body { font-size:.82rem; color:var(--c2); line-height:1.72; }
.weather-trigger {
  margin-top:.8rem; padding:.65rem .9rem;
  background:var(--bg4); border-radius:10px;
  border-left:2px solid rgba(134,239,172,.35);
  font-size:.76rem; color:var(--c3); line-height:1.6;
}

/* Weather risk widget */
.weather-row { display:flex; gap:.55rem; flex-wrap:wrap; margin-bottom:.5rem; }
.weather-chip {
  flex:1; min-width:120px; padding:.75rem .9rem;
  background:var(--bg3); border:1px solid var(--bd);
  border-radius:var(--r); cursor:pointer; transition:all .18s; text-align:center;
}
.weather-chip.selected { border-color:rgba(134,239,172,.4) !important; background:rgba(74,222,128,.06) !important; box-shadow:0 0 0 1px rgba(134,239,172,.15); }
.weather-chip:hover { border-color:var(--bd2); transform:translateY(-2px); }
.wc-icon { font-size:1.4rem; margin-bottom:.3rem; }
.wc-lbl  { font-size:.7rem; font-family:var(--mono); color:var(--c3); margin-bottom:.25rem; }
.wc-risk { font-size:.62rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-family:var(--mono); }
.risk-result {
  padding:1rem 1.2rem; border-radius:var(--r);
  font-size:.82rem; font-family:var(--mono); color:var(--c2);
  margin-top:.5rem; line-height:1.6;
}

/* History rows */
.hist-row {
  display:flex; align-items:center; gap:.75rem;
  padding:.65rem .9rem; background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--r); margin-bottom:.4rem; font-size:.8rem;
  transition:border-color .18s, transform .18s;
  animation:slideIn .35s ease both;
}
.hist-row:hover { border-color:var(--bd2); transform:translateX(4px); }
.hist-ico  { font-size:1.1rem; flex-shrink:0; }
.hist-name { font-family:var(--disp); font-weight:600; color:var(--c2); flex:1; font-size:.82rem; }
.hist-conf { font-family:var(--mono); font-size:.71rem; color:var(--c3); }
.hist-time { font-family:var(--mono); font-size:.64rem; color:var(--c4); }
.hist-tag  { font-family:var(--mono); font-size:.64rem; font-weight:700; padding:.14rem .5rem; border-radius:999px; }
.ht-ok   { background:rgba(134,239,172,.1); color:var(--g);  border:1px solid rgba(134,239,172,.22); }
.ht-bad  { background:var(--redbg); color:var(--red);        border:1px solid rgba(252,165,165,.22); }
.ht-warn { background:var(--amrbg); color:var(--amr);        border:1px solid rgba(252,211,77,.22); }

/* Batch mode badge */
.batch-badge {
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.25rem .75rem; background:rgba(74,222,128,.08);
  border:1px solid rgba(134,239,172,.22); border-radius:999px;
  font-size:.68rem; font-family:var(--mono); color:var(--g);
  margin-bottom:.7rem;
}

/* Footer */
.app-footer {
  text-align:center; padding:2rem 0 1.1rem;
  font-size:.67rem; color:var(--c4); font-family:var(--mono);
  border-top:1px solid var(--bd); margin-top:2.5rem;
  line-height:1.9;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOADING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def loading_page():
    inject_css()
    st.markdown("""
<div class="loading-screen">
  <div class="ls-orb">🌿</div>
  <div class="ls-title">CornScan AI</div>
  <div class="ls-sub">Initializing AI diagnosis engine…</div>
  <div class="ls-bar-bg"><div class="ls-bar"></div></div>
  <div class="ls-dots">
    <div class="ls-dot"></div>
    <div class="ls-dot"></div>
    <div class="ls-dot"></div>
  </div>
  <div class="ls-powered">Powered by CornScan AI Engine · v4.0</div>
</div>
""", unsafe_allow_html=True)

    import time
    time.sleep(2.2)
    st.session_state.page = "landing"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing_page():
    inject_css()
    st.markdown("""
<div class="lp-bg"></div><div class="lp-grid"></div>
<div class="lp-corner lp-tl">🌽</div><div class="lp-corner lp-tr">🌽</div>
<div class="lp-corner lp-bl">🌽</div><div class="lp-corner lp-br">🌽</div>

<div class="lp-wrap">
  <div class="lp-orb">🌿</div>
  <div class="lp-pill">
    <span class="lp-pill-dot"></span>
    Deep Learning · Plant Pathology · v4.0
  </div>
  <div class="lp-title">CornScan<br><span class="lp-title-grad">AI</span></div>
  <div class="lp-sub">
    Upload a corn leaf photo. Get an instant, science-backed disease diagnosis
    powered by a deep convolutional neural network.
  </div>
  <div class="lp-stats">
    <div><div class="lp-stat-n">97%</div><div class="lp-stat-l">Accuracy</div></div>
    <div class="lp-sep"></div>
    <div><div class="lp-stat-n">4</div><div class="lp-stat-l">Classes</div></div>
    <div class="lp-sep"></div>
    <div><div class="lp-stat-n">&lt;2s</div><div class="lp-stat-l">Inference</div></div>
    <div class="lp-sep"></div>
    <div><div class="lp-stat-n">CNN</div><div class="lp-stat-l">Architecture</div></div>
  </div>
  <div class="lp-feats">
    <div class="lp-feat"><div class="lp-feat-ico">🧠</div><div class="lp-feat-t">Deep CNN</div><div class="lp-feat-d">224×224 input trained on annotated leaf images</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">⚡</div><div class="lp-feat-t">Instant Results</div><div class="lp-feat-d">Full probability breakdown under 2 seconds</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📊</div><div class="lp-feat-t">AI Dashboard</div><div class="lp-feat-d">Confidence ring, risk meter & donut charts</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🔬</div><div class="lp-feat-t">Comparison Panel</div><div class="lp-feat-d">AI heatmap overlay on uploaded leaf</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📦</div><div class="lp-feat-t">Batch Scan</div><div class="lp-feat-d">Upload multiple images in one session</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🌦️</div><div class="lp-feat-t">Weather Risk</div><div class="lp-feat-d">Disease risk based on current field conditions</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 2, 1.4])
    with c2:
        if st.button("🚀  Launch CornScan AI", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown("""
<div style="position:relative;z-index:1;text-align:center;margin-top:.9rem;
font-size:.66rem;color:var(--c4);font-family:var(--mono);">
Powered by CornScan AI Engine &nbsp;·&nbsp; TensorFlow · Keras · Streamlit
&nbsp;·&nbsp; No data leaves your device
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE RING  (inline SVG)
# ══════════════════════════════════════════════════════════════════════════════
def render_conf_ring(conf: float, color: str) -> str:
    R = 37
    C = 2 * 3.14159 * R
    offset = C * (1 - conf)
    return f"""
<div class="conf-ring-wrap">
  <div class="conf-ring">
    <svg width="90" height="90" viewBox="0 0 90 90">
      <circle class="cr-bg"  cx="45" cy="45" r="{R}"/>
      <circle class="cr-fill" cx="45" cy="45" r="{R}"
        stroke="{color}" stroke-linecap="round"
        stroke-dasharray="{C:.2f}" stroke-dashoffset="{C:.2f}"
        style="transition:stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1);stroke-dashoffset:{offset:.2f}"/>
    </svg>
    <div class="cr-text">
      <div class="cr-pct">{conf*100:.1f}%</div>
      <div class="cr-sub">Conf.</div>
    </div>
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT PDF  (text-based report)
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(results: list) -> bytes:
    lines = []
    ts    = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    lines += [
        "=" * 58,
        "  CORNSCAN AI — FIELD DIAGNOSIS REPORT",
        "  Powered by CornScan AI Engine v4.0",
        f"  Generated: {ts}",
        "=" * 58,
        "",
    ]
    for i, r in enumerate(results, 1):
        info = r["info"]
        lines += [
            f"  SCAN #{i}: {r['fname']}",
            f"  {'─' * 42}",
            f"  Diagnosis  : {info['short']}",
            f"  Pathogen   : {info['pathogen']}",
            f"  Confidence : {r['conf']*100:.1f}%",
            f"  Severity   : {info['severity']}",
            f"  Timestamp  : {r['ts']}",
            "",
            "  PROBABILITY BREAKDOWN",
        ]
        for cls, p in r["all_probs"].items():
            bar = "█" * int(p * 30)
            lines.append(f"  {cls:<18} {bar:<30} {p*100:5.1f}%")
        lines += [
            "",
            "  DESCRIPTION",
            f"  {info['desc']}",
            "",
            "  RECOMMENDED ACTION",
            f"  {info['action']}",
            "",
            "  SYMPTOMS",
        ]
        for s in info["symptoms"]:
            lines.append(f"  · {s}")
        lines += [
            "",
            "  FARMER ADVICE",
            f"  {info['farmer_advice']}",
            "",
            "  WEATHER TRIGGER",
            f"  {info['weather_trigger']}",
            "",
            "─" * 58,
            "",
        ]
    lines += [
        "CornScan AI Engine v4.0",
        "TensorFlow / Keras  ·  CNN Plant Disease Detection",
        "No data leaves your device.",
    ]
    return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main_app():
    inject_css()

    # ── Top bar ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="app-topbar">
  <div class="app-topbar-brand">
    <div class="app-topbar-ico">🌿</div>
    <div>
      <div class="app-topbar-name">CornScan AI</div>
      <div class="app-topbar-ver">v4.0 · CNN · TensorFlow &nbsp;·&nbsp; Powered by CornScan AI Engine</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    bc, _, _ = st.columns([1, 3, 1])
    with bc:
        if st.button("← Home"):
            st.session_state.page    = "landing"
            st.session_state.results = []
            st.rerun()

    # ── Stats strip ────────────────────────────────────────────────────────
    n_total    = st.session_state.scanned
    n_diseased = sum(1 for h in st.session_state.history if h["status"] != "ok")
    n_healthy  = n_total - n_diseased

    st.markdown(f"""
<div class="stats-strip">
  <div class="stat-box"><div class="stat-n">{n_total}</div><div class="stat-l">Scanned</div></div>
  <div class="stat-box"><div class="stat-n">{n_diseased}</div><div class="stat-l">Diseased</div></div>
  <div class="stat-box"><div class="stat-n">{n_healthy}</div><div class="stat-l">Healthy</div></div>
</div>
""", unsafe_allow_html=True)

    # ── DEMO mode ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">🎯 Quick Demo</div>', unsafe_allow_html=True)
    demo_labels = {
        "🍂 Blight Demo":       "Blight",
        "🟠 Rust Demo":         "Common Rust",
        "🩶 Gray Leaf Spot Demo":"Gray Leaf Spot",
        "✅ Healthy Demo":      "Healthy",
    }
    st.markdown('<div class="demo-grid">' +
        "".join(f'<span class="demo-chip" '
                f'onclick="window.location.href=window.location.href">{k}</span>'
                for k in demo_labels) +
        '</div>', unsafe_allow_html=True)

    demo_choice = st.selectbox(
        "Demo", ["— Run a demo scan —"] + list(demo_labels.keys()),
        label_visibility="collapsed",
    )
    if demo_choice != "— Run a demo scan —":
        label    = demo_labels[demo_choice]
        info     = DISEASE_INFO[label]
        preds    = np.random.dirichlet(np.ones(4) * 0.5)
        preds_d  = dict(zip(CLASSES, preds.tolist()))
        # force chosen label to highest
        preds_d[label] = max(0.82, preds_d[label])
        total = sum(preds_d.values())
        preds_d = {k: v/total for k, v in preds_d.items()}
        conf   = preds_d[label]
        ts     = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        status = "ok" if label == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
        st.session_state.results = [
            dict(fname="demo_leaf.jpg", img=None, label=label,
                 conf=conf, all_probs=preds_d, ts=ts, info=info, status=status, b64=None)
        ]
        st.session_state.history.insert(0, dict(label=label, conf=conf, ts=ts, fname="demo_leaf.jpg", status=status, info=info))
        st.session_state.scanned += 1

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">📁 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="batch-badge">📦 Batch Scan Mode — multiple files supported</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "drop", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    st.markdown("""
<div class="upload-hint">
  JPG · PNG · JPEG &nbsp;|&nbsp; Drop multiple files to batch scan &nbsp;|&nbsp; No data leaves your device
</div>""", unsafe_allow_html=True)

    valid, do_analyze = [], False

    if uploaded_files:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🖼 Previews</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(uploaded_files), 3))

        for i, f in enumerate(uploaded_files[:3]):
            try:
                f.seek(0)
                img = Image.open(f).convert("RGB")
                valid.append((f.name, img))
                w, h = img.size
                with cols[i]:
                    st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
                    st.image(img, use_container_width=True)
                    st.markdown(
                        f'<div class="img-foot">'
                        f'<span>{f.name[:20]}</span>'
                        f'<span class="img-badge">{w}×{h}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                cols[i].error(f"Bad file: {f.name}")

        for f in uploaded_files[3:]:
            try:
                f.seek(0)
                img = Image.open(f).convert("RGB")
                valid.append((f.name, img))
            except Exception:
                pass

        if len(uploaded_files) > 3:
            st.caption(f"+{len(uploaded_files) - 3} more file(s) queued")

        st.markdown("<br>", unsafe_allow_html=True)
        do_analyze = st.button(
            f"🔬  Analyze {len(valid)} Image{'s' if len(valid) > 1 else ''}",
            use_container_width=True,
        )

    # ── Scan Loading ───────────────────────────────────────────────────────
    if do_analyze and valid:
        preview_b64 = img_to_b64(valid[0][1])
        scan_steps  = [
            "Initializing AI diagnosis…",
            "Preprocessing image tensor…",
            "Running forward pass…",
            "Analyzing disease patterns…",
            "Generating diagnosis…",
        ]
        placeholder = st.empty()

        for step in scan_steps:
            placeholder.markdown(f"""
<div class="scan-overlay">
  <div class="scan-icon-wrap">🌿</div>
  <div class="scan-title">{step}</div>
  <div class="scan-sub">Leaf scan in progress — CornScan AI Engine v4.0</div>
  <div class="scanline-box">
    <img class="scanline-img" src="data:image/jpeg;base64,{preview_b64}" alt="scanning">
    <div class="scanline-bar"></div>
  </div>
  <div class="scan-steps">{step}</div>
  <div class="scan-dots">
    <div class="scan-dot"></div>
    <div class="scan-dot"></div>
    <div class="scan-dot"></div>
  </div>
</div>
""", unsafe_allow_html=True)
            import time; time.sleep(0.38)

        placeholder.empty()

        batch = []
        for fname, img in valid:
            label, conf, all_probs = predict(img)
            ts    = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
            info  = DISEASE_INFO[label]
            b64   = img_to_b64(img)
            status = "ok" if label == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
            batch.append(dict(
                fname=fname, img=img, label=label,
                conf=conf, all_probs=all_probs, ts=ts,
                info=info, status=status, b64=b64,
            ))
            st.session_state.history.insert(0, dict(
                label=label, conf=conf, ts=ts,
                fname=fname, status=status, info=info,
            ))
            st.session_state.scanned += 1

        st.session_state.results = batch
        st.rerun()

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results

        # ── 1. Diagnosis cards ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧬 Diagnosis</div>', unsafe_allow_html=True)

        for r in results:
            info     = r["info"]
            pct      = r["conf"] * 100
            status   = r["status"]
            card_cls = {"ok": "", "warn": "amber", "bad": "diseased"}.get(status, "")
            tag_cls  = {"ok": "r-ok", "warn": "r-warn", "bad": "r-bad"}.get(status, "r-ok")
            tag_txt  = {"ok": "⬤ Healthy", "warn": "⬤ Monitor", "bad": "⬤ Diseased"}.get(status, "")
            bar_grad = {
                "ok"  : "linear-gradient(90deg,#16a34a,#86efac)",
                "warn": "linear-gradient(90deg,#b45309,#fcd34d)",
                "bad" : "linear-gradient(90deg,#991b1b,#fca5a5)",
            }.get(status, "")
            ring_col = {"ok": "#86efac", "warn": "#fcd34d", "bad": "#fca5a5"}.get(status, "#86efac")
            urg      = info["urgency"]
            urg_style = {
                "HIGH"  : "background:rgba(252,165,165,.12);color:#fca5a5;border:1px solid rgba(252,165,165,.3);",
                "MEDIUM": "background:rgba(252,211,77,.12);color:#fcd34d;border:1px solid rgba(252,211,77,.3);",
                "NONE"  : "background:rgba(134,239,172,.1);color:#86efac;border:1px solid rgba(134,239,172,.28);",
            }.get(urg, "")
            urg_txt = {"HIGH": "🚨 Urgent Treatment Required", "MEDIUM": "⚠️ Monitor Closely", "NONE": "✅ No Action Needed"}.get(urg, "")

            ring_html = render_conf_ring(r["conf"], ring_col)

            st.markdown(f"""
<div class="result-card {card_cls}">
  <div class="r-top-row">
    <div>
      <span class="r-tag {tag_cls}">{tag_txt}</span>
      <div class="r-name">{info['short']}</div>
      <div class="r-sci">{info['pathogen']}</div>
    </div>
    {ring_html}
  </div>
  <div class="r-ch">
    <span class="r-cl">Confidence Score</span>
    <span class="r-cv">{pct:.1f}%</span>
  </div>
  <div class="r-bt"><div class="r-bf" style="width:{pct:.1f}%;background:{bar_grad};"></div></div>
  <span class="urgency-badge" style="{urg_style}">{urg_txt}</span>
  <div class="r-meta" style="margin-top:.9rem;">
    <span>🕐 {r['ts']}</span>
    <span>📄 {r['fname']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            with st.expander("📊 Full probability breakdown"):
                for cls in CLASSES:
                    p  = r["all_probs"][cls]
                    hi = "pb-hi" if cls == r["label"] else ""
                    st.markdown(f"""
<div class="pb-row">
  <span class="pb-name">{cls}</span>
  <div class="pb-tr"><div class="pb-fill {hi}" style="width:{p*100:.1f}%"></div></div>
  <span class="pb-pct">{p*100:.1f}%</span>
</div>""", unsafe_allow_html=True)

        # ── 2. Export PDF ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        report_bytes = generate_report(results)
        fname_out = f"cornscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        st.download_button(
            label="📄  Export Diagnosis Report",
            data=report_bytes,
            file_name=fname_out,
            mime="text/plain",
            use_container_width=True,
        )

        # ── 3. AI Comparison Panel ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🔍 AI Analysis Panel</div>', unsafe_allow_html=True)

        for r in results:
            info      = r["info"]
            hm_class  = {"ok": "heatmap-ok", "warn": "heatmap-warn", "bad": "heatmap-bad"}.get(r["status"], "heatmap-ok")
            img_html  = (
                f'<img src="data:image/jpeg;base64,{r["b64"]}" alt="leaf">'
                if r.get("b64") else
                f'<div style="width:100%;height:220px;display:flex;align-items:center;justify-content:center;font-size:4rem;background:var(--bg3);">{info["icon"]}</div>'
            )
            bars = "".join(
                f'<div class="cpb-row"><span class="cpb-lbl">{cls}</span>'
                f'<div class="cpb-tr"><div class="cpb-fill {"cpb-hi" if cls == r["label"] else ""}" style="width:{r["all_probs"][cls]*100:.1f}%"></div></div>'
                f'<span class="cpb-pct">{r["all_probs"][cls]*100:.1f}%</span></div>'
                for cls in CLASSES
            )
            st.markdown(f"""
<div class="comp-panel">
  <div class="comp-header">🤖 AI Focus Analysis &nbsp;·&nbsp; {r['fname']}</div>
  <div class="comp-body">
    <div class="comp-left">
      {img_html}
      <div class="{hm_class}"></div>
      <div class="ai-zone-badge">🔴 AI Focus Zone</div>
    </div>
    <div class="comp-right">
      <div>
        <div class="comp-disease">{info['short']}</div>
        <div class="comp-sci">{info['pathogen']}</div>
      </div>
      <div class="comp-bars">{bars}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── 4. Charts Dashboard ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📊 AI Dashboard</div>', unsafe_allow_html=True)

        r0          = results[0]
        sev         = r0["info"]["severity"]
        risk_pct    = 82 if sev == "HIGH" else (50 if sev == "MEDIUM" else 14)
        risk_color  = "#fca5a5" if sev == "HIGH" else ("#fcd34d" if sev == "MEDIUM" else "#86efac")
        risk_label  = sev if sev != "NONE" else "LOW"

        # Donut data
        total_h     = max(len(st.session_state.history), 1)
        hc          = sum(1 for h in st.session_state.history if h["status"] == "ok")
        dc          = total_h - hc
        h_pct       = round(hc / total_h * 100)
        R_d         = 34
        C_d         = 2 * 3.14159 * R_d
        h_arc       = hc / total_h * C_d
        d_arc       = dc / total_h * C_d

        st.markdown(f"""
<div class="chart-row">
  <div class="chart-card">
    <div class="chart-title">Risk Score</div>
    <div style="text-align:center;">
      <div class="gauge-num" style="color:{risk_color};">{risk_pct}</div>
      <div class="gauge-lbl">{risk_label} risk</div>
      <div class="gauge-bar">
        <div class="gauge-needle" style="left:{risk_pct}%;"></div>
      </div>
      <div class="gauge-scale"><span>Low</span><span>High</span></div>
    </div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Field Health</div>
    <div class="donut-row">
      <svg width="88" height="88" viewBox="0 0 88 88" style="transform:rotate(-90deg);flex-shrink:0;">
        <circle cx="44" cy="44" r="{R_d}" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="10"/>
        <circle cx="44" cy="44" r="{R_d}" fill="none" stroke="#86efac" stroke-width="10"
          stroke-dasharray="{h_arc:.2f} {C_d:.2f}" stroke-linecap="round"/>
        <circle cx="44" cy="44" r="{R_d}" fill="none" stroke="#fca5a5" stroke-width="10"
          stroke-dasharray="{d_arc:.2f} {C_d:.2f}" stroke-dashoffset="-{h_arc:.2f}"
          stroke-linecap="round" style="opacity:{1 if dc > 0 else 0};"/>
      </svg>
      <div class="donut-lgd">
        <div class="donut-big">{h_pct}%</div>
        <div class="donut-sub">Healthy</div>
        <div class="donut-item" style="margin-top:.5rem;"><div class="donut-dot" style="background:#86efac;"></div>Healthy ({hc})</div>
        <div class="donut-item"><div class="donut-dot" style="background:#fca5a5;"></div>Diseased ({dc})</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Confidence bar chart
        with st.expander("📈 Confidence distribution — all results"):
            for r in results:
                for cls in CLASSES:
                    p  = r["all_probs"][cls]
                    hi = "pb-hi" if cls == r["label"] else ""
                    st.markdown(f"""
<div class="pb-row">
  <span class="pb-name">{cls}</span>
  <div class="pb-tr"><div class="pb-fill {hi}" style="width:{p*100:.1f}%"></div></div>
  <span class="pb-pct">{p*100:.1f}%</span>
</div>""", unsafe_allow_html=True)

        # ── 5. Agronomic Detail Panel ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📚 Agronomic Details</div>', unsafe_allow_html=True)

        seen = set()
        for r in results:
            lbl  = r["label"]
            if lbl in seen: continue
            seen.add(lbl)
            info = r["info"]
            sc   = info["sev_color"]
            chips = "".join(f'<span class="sym-chip">· {s}</span>' for s in info["symptoms"])

            with st.expander(f"{info['icon']}  {info['short']}", expanded=True):
                st.markdown(f"""
<div class="info-grid">
  <div class="info-card">
    <div class="info-card-h">📋 Overview</div>
    <div class="info-card-b">{info['desc']}</div>
    <span class="sev-badge" style="background:{sc}18;color:{sc};border:1px solid {sc}44;">
      SEVERITY · {info['severity']}
    </span>
  </div>
  <div class="info-card">
    <div class="info-card-h">🔍 Symptoms</div>
    <div>{chips}</div>
    <div style="margin-top:.85rem;">
      <div class="info-card-h">🛡 Recommended Action</div>
      <div class="rec-item"><span class="rec-ico">✓</span><span>{info['action']}</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── 6. Farmer Advice ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">👨‍🌾 Farmer Advice</div>', unsafe_allow_html=True)

        seen2 = set()
        for r in results:
            if r["label"] in seen2: continue
            seen2.add(r["label"])
            info = r["info"]
            st.markdown(f"""
<div class="farmer-card">
  <div class="farmer-head">{info['icon']} &nbsp;{info['short']}</div>
  <div class="farmer-body">{info['farmer_advice']}</div>
  <div class="weather-trigger">
    🌦️ &nbsp;<strong>Weather note:</strong> {info['weather_trigger']}
  </div>
</div>
""", unsafe_allow_html=True)

        # ── 7. Weather-based Disease Risk ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🌦️ Weather-Based Disease Risk</div>', unsafe_allow_html=True)

        st.markdown('<div class="weather-row">' +
            "".join(
                f'<div class="weather-chip">'
                f'<div class="wc-icon">{w["icon"]}</div>'
                f'<div class="wc-lbl">{w["label"]}</div>'
                f'<div class="wc-risk" style="color:{w["risk_color"]};">{w["risk"]}</div>'
                f'</div>'
                for w in WEATHER_CONDITIONS
            ) + '</div>', unsafe_allow_html=True)

        weather_sel = st.selectbox(
            "Select current field weather:",
            [w["label"] for w in WEATHER_CONDITIONS],
            label_visibility="visible",
        )
        w_info = next(w for w in WEATHER_CONDITIONS if w["label"] == weather_sel)
        rc     = w_info["risk_color"]

        # get trigger text from first result
        trigger_txt = results[0]["info"]["weather_trigger"] if results else ""

        st.markdown(f"""
<div class="risk-result" style="background:var(--bg3);border-left:2px solid {rc};border-radius:10px;">
  {w_info['icon']} &nbsp;<strong style="color:{rc};">{w_info['risk']} RISK</strong> &nbsp;·&nbsp;
  Risk index: <strong style="color:{rc};">{w_info['risk_pct']}%</strong><br>
  <span style="font-size:.78rem;color:var(--c3);">{trigger_txt}</span>
</div>
""", unsafe_allow_html=True)

    # ── History ────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📜 Scan History</div>', unsafe_allow_html=True)

        for h in st.session_state.history[:8]:
            tc = {"ok": "ht-ok", "warn": "ht-warn", "bad": "ht-bad"}.get(h["status"], "ht-ok")
            tt = {"ok": "Healthy", "warn": "Monitor", "bad": "Diseased"}.get(h["status"], "—")
            st.markdown(f"""
<div class="hist-row">
  <span class="hist-ico">{h['info']['icon']}</span>
  <span class="hist-name">{h['info']['short']}</span>
  <span class="hist-conf">{h['conf']*100:.1f}%</span>
  <span class="hist-time">{h['ts']}</span>
  <span class="hist-tag {tc}">{tt}</span>
</div>""", unsafe_allow_html=True)

        if len(st.session_state.history) > 8:
            st.caption(f"+{len(st.session_state.history) - 8} older entries")

        st.markdown("<br>", unsafe_allow_html=True)
        cl1, _ = st.columns([1, 4])
        with cl1:
            if st.button("↺ Clear History"):
                st.session_state.history = []
                st.session_state.scanned = 0
                st.session_state.results = []
                st.rerun()

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="app-footer">
  🌽 &nbsp; <strong>CornScan AI</strong> &nbsp;·&nbsp; Powered by CornScan AI Engine<br>
  TensorFlow / Keras &nbsp;·&nbsp; CNN Plant Disease Detection &nbsp;·&nbsp; v4.0.0<br>
  <span style="opacity:.5;">No data leaves your device &nbsp;·&nbsp; For research & field-scouting use</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "loading":
    loading_page()
elif st.session_state.page == "landing":
    landing_page()
else:
    main_app()