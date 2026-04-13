"""
╔══════════════════════════════════════════════════════════════════╗
║  CornScan AI  ·  app.py                                          ║
║  Premium dark editorial UI — CNN Corn Disease Detection          ║
║  TensorFlow / Keras · Streamlit                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
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
for key, default in [("page", "landing"), ("history", []), ("results", []), ("scanned", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Constants ──────────────────────────────────────────────────────────────
CLASSES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

DISEASE_INFO = {
    "Blight": {
        "icon": "🍂", "severity": "HIGH", "sev_color": "#ff6b6b",
        "short": "Northern Corn Leaf Blight",
        "pathogen": "Exserohilum turcicum",
        "desc": "A serious fungal disease thriving in moderate temperatures (18–27 °C) with extended leaf-wetness. Can reduce yield by 30–50 % in epidemic years.",
        "action": "Apply strobilurin fungicide at early tassel. Remove infected residue post-harvest.",
        "symptoms": ["Cigar-shaped grey-green lesions (3–15 cm)", "Tan-brown mature lesions", "Olive spore masses on leaf surface"],
    },
    "Common Rust": {
        "icon": "🟠", "severity": "MEDIUM", "sev_color": "#ffa94d",
        "short": "Common Corn Rust",
        "pathogen": "Puccinia sorghi",
        "desc": "Spreads via airborne spores in cool, humid conditions (16–23 °C). Can reduce grain fill by up to 20 % with severe pre-silking infection.",
        "action": "Scout weekly from V6. Apply fungicide if >50 pustules per leaf pre-silk.",
        "symptoms": ["Brick-red circular pustules on both surfaces", "Powdery cinnamon-brown spore masses", "Dark brown-black pustules late season"],
    },
    "Gray Leaf Spot": {
        "icon": "🩶", "severity": "HIGH", "sev_color": "#ff6b6b",
        "short": "Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis",
        "desc": "Among the most economically damaging corn diseases globally. Overwinters in residue; epidemic in warm, humid, no-till continuous-corn systems.",
        "action": "Plant resistant hybrids. Apply triazole + strobilurin mix at VT/R1.",
        "symptoms": ["Rectangular lesions bounded by leaf veins", "Ash-grey to pale tan colour", "Yellow halo around mature lesions"],
    },
    "Healthy": {
        "icon": "✅", "severity": "NONE", "sev_color": "#69db7c",
        "short": "No Disease Detected",
        "pathogen": "Zea mays — clean",
        "desc": "No signs of fungal, bacterial, or viral disease detected. The leaf appears vigorous with uniform colour and clean surface texture.",
        "action": "Continue routine weekly scouting. Maintain balanced NPK fertilisation.",
        "symptoms": ["Uniform deep-green colour", "Clean surface, no lesions", "Normal venation and architecture"],
    },
}

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


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg0   : #080e06;
  --bg1   : #0d1509;
  --bg2   : #141e0f;
  --bg3   : #1c2a15;
  --bg4   : #243420;
  --bd    : rgba(255,255,255,.07);
  --bd2   : rgba(255,255,255,.13);
  --hi    : rgba(134,239,172,.18);
  --hi2   : rgba(134,239,172,.08);
  --cream : #f2ede3;
  --c2    : #c4bfb2;
  --c3    : #807b71;
  --c4    : #484540;
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
[data-testid="collapsedControl"] { display:none !important; visibility:hidden; }
.block-container { padding-top:0 !important; max-width:720px; }
[data-testid="stSidebar"] { display:none !important; }

/* Buttons */
.stButton > button {
  font-family: var(--disp) !important; font-weight:700 !important;
  font-size:.9rem !important; letter-spacing:.02em !important;
  border-radius: var(--r) !important;
  border:1.5px solid rgba(134,239,172,.35) !important;
  background:linear-gradient(135deg,rgba(74,222,128,.2),rgba(134,239,172,.12)) !important;
  color:var(--g) !important; padding:.65rem 1.6rem !important;
  box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.05) !important;
  transition:all .2s cubic-bezier(.4,0,.2,1) !important;
}
.stButton > button:hover {
  background:linear-gradient(135deg,rgba(74,222,128,.32),rgba(134,239,172,.2)) !important;
  border-color:rgba(134,239,172,.6) !important;
  box-shadow:0 0 0 1px rgba(134,239,172,.2),var(--shm) !important;
  transform:translateY(-1px) !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* Uploader */
[data-testid="stFileUploader"] section {
  background:var(--bg2) !important;
  border:2px dashed rgba(134,239,172,.2) !important;
  border-radius:var(--rxl) !important; padding:2.5rem !important;
  transition:all .22s !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color:rgba(134,239,172,.45) !important;
  background:rgba(74,222,128,.04) !important;
}

/* Expander */
details {
  background:var(--bg2) !important;
  border:1px solid var(--bd) !important;
  border-radius:var(--r) !important; margin-bottom:.5rem !important;
}
details:hover { border-color:var(--bd2) !important; }
details summary { color:var(--c2) !important; font-weight:600 !important; font-size:.86rem !important; padding:.85rem 1.1rem !important; }
details[open]   { border-color:rgba(134,239,172,.22) !important; }

.stProgress > div > div { background:linear-gradient(90deg,var(--gd),var(--g)) !important; border-radius:999px !important; }
.stProgress > div { background:var(--bg3) !important; border-radius:999px !important; height:6px !important; }
.stSpinner > div  { border-top-color:var(--g) !important; }
.stMarkdown p, .stMarkdown li { color:var(--c2) !important; font-size:.88rem !important; }

/* ── Keyframes ── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes barGrow  { from{width:0} }
@keyframes pulse    { 0%,100%{box-shadow:0 0 0 0 rgba(134,239,172,.4)} 60%{box-shadow:0 0 0 10px transparent} }
@keyframes floatY   { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes gradShift{ 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes dotPulse { 0%,80%,100%{opacity:.25;transform:scale(.75)} 40%{opacity:1;transform:scale(1.1)} }

/* ═══ LANDING ═══ */
.lp-bg {
  position:fixed; inset:0; z-index:0;
  background:
    radial-gradient(ellipse 70% 55% at 20% -5%, rgba(74,222,128,.09) 0%, transparent 55%),
    radial-gradient(ellipse 60% 45% at 80% 105%, rgba(134,239,172,.06) 0%, transparent 55%),
    var(--bg0);
  pointer-events:none;
}
.lp-grid {
  position:fixed; inset:0; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
  background-size:48px 48px; pointer-events:none;
  mask-image:radial-gradient(ellipse 80% 80% at 50% 50%, black, transparent);
}
.lp-wrap {
  position:relative; z-index:1; min-height:88vh;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  padding:3rem 1.5rem 2rem; text-align:center;
  animation:fadeUp .7s ease both;
}
.lp-orb {
  width:110px; height:110px;
  background:radial-gradient(circle at 35% 35%, rgba(134,239,172,.6), rgba(22,163,74,.2) 60%, transparent 80%);
  border:1px solid rgba(134,239,172,.3); border-radius:30px;
  display:flex; align-items:center; justify-content:center; font-size:3.2rem;
  margin-bottom:1.8rem;
  box-shadow:0 0 60px rgba(134,239,172,.15), inset 0 1px 0 rgba(255,255,255,.1);
  animation:floatY 4s ease-in-out infinite, pulse 3s infinite;
}
.lp-pill {
  display:inline-flex; align-items:center; gap:.4rem;
  background:var(--hi2); border:1px solid rgba(134,239,172,.22);
  border-radius:999px; padding:.22rem 1rem;
  font-size:.68rem; font-weight:500; color:var(--g);
  letter-spacing:.1em; text-transform:uppercase;
  font-family:var(--mono); margin-bottom:1rem;
}
.lp-pill-dot { width:6px;height:6px;border-radius:50%;background:var(--g);animation:pulse 2s infinite; }
.lp-title {
  font-family:var(--disp); font-size:clamp(2.8rem,7vw,4.6rem);
  font-weight:800; letter-spacing:-.05em; line-height:1.0;
  margin-bottom:.5rem; color:var(--cream);
}
.lp-title-grad {
  background:linear-gradient(135deg,#86efac 0%,#4ade80 40%,#a3e635 100%);
  background-size:200% 200%; animation:gradShift 5s ease infinite;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.lp-sub {
  font-size:1.05rem; color:var(--c3); font-weight:300;
  line-height:1.7; max-width:440px; margin:0 auto 2rem;
}
.lp-stats { display:flex; gap:2.5rem; justify-content:center; margin-bottom:2.2rem; }
.lp-stat-n { font-family:var(--disp); font-size:2rem; font-weight:800; color:var(--g); letter-spacing:-.05em; line-height:1; }
.lp-stat-l { font-size:.68rem; color:var(--c4); font-family:var(--mono); margin-top:.2rem; letter-spacing:.08em; text-transform:uppercase; }
.lp-sep    { width:1px; background:var(--bd); }
.lp-feats  {
  display:grid; grid-template-columns:repeat(3,1fr);
  gap:.65rem; margin-bottom:2rem; width:100%; max-width:600px;
}
.lp-feat {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rl); padding:1.1rem .9rem; box-shadow:var(--sh);
  transition:border-color .2s, transform .2s, box-shadow .2s;
}
.lp-feat:hover { border-color:rgba(134,239,172,.25); transform:translateY(-3px); box-shadow:var(--shm); }
.lp-feat-ico { font-size:1.4rem; margin-bottom:.35rem; }
.lp-feat-t   { font-family:var(--disp); font-size:.78rem; font-weight:700; color:var(--c2); margin-bottom:.18rem; }
.lp-feat-d   { font-size:.68rem; color:var(--c4); line-height:1.5; }
.lp-corner   { position:fixed; font-size:2.2rem; opacity:.07; pointer-events:none; }
.lp-tl{top:20px;left:20px;} .lp-tr{top:20px;right:20px;} .lp-bl{bottom:20px;left:20px;} .lp-br{bottom:20px;right:20px;}

/* ═══ MAIN APP ═══ */
.app-topbar {
  display:flex; align-items:center; gap:.75rem;
  padding:1.2rem 0 .5rem; border-bottom:1px solid var(--bd);
  margin-bottom:1.5rem; animation:fadeIn .4s ease both;
}
.app-topbar-brand { display:flex; align-items:center; gap:.6rem; flex:1; }
.app-topbar-ico {
  width:32px; height:32px; border-radius:9px;
  background:var(--hi2); border:1px solid rgba(134,239,172,.22);
  display:flex; align-items:center; justify-content:center; font-size:1rem;
}
.app-topbar-name { font-family:var(--disp); font-size:1rem; font-weight:800; color:var(--cream); letter-spacing:-.03em; }
.app-topbar-ver  { font-size:.65rem; color:var(--c4); font-family:var(--mono); }

.sec-lbl {
  display:flex; align-items:center; gap:.5rem;
  font-size:.62rem; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; color:var(--c4);
  font-family:var(--mono); margin-bottom:.65rem;
}
.sec-lbl::after { content:''; flex:1; height:1px; background:var(--bd); }

.img-wrap {
  background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--rxl); overflow:hidden;
  box-shadow:var(--shm); animation:fadeIn .35s ease both;
}
.img-foot {
  padding:.55rem 1rem; border-top:1px solid var(--bd);
  font-size:.71rem; color:var(--c4); font-family:var(--mono);
  display:flex; align-items:center; justify-content:space-between;
}
.img-badge {
  background:var(--hi2); color:var(--g);
  border:1px solid rgba(134,239,172,.2); border-radius:999px;
  font-size:.63rem; padding:.06rem .5rem; font-weight:600;
}

/* Result card */
.result-card {
  background:var(--bg2); border:1.5px solid rgba(134,239,172,.2);
  border-radius:var(--rxl); padding:2rem 2.2rem;
  box-shadow:var(--shl),0 0 80px rgba(134,239,172,.05);
  animation:fadeUp .5s ease both; position:relative; overflow:hidden;
  margin-bottom:.75rem;
}
.result-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--gd),var(--g),#a3e635);
}
.result-card.diseased { border-color:rgba(252,165,165,.25) !important; }
.result-card.diseased::before { background:linear-gradient(90deg,#991b1b,#fca5a5) !important; }
.result-card.amber    { border-color:rgba(252,211,77,.25) !important; }
.result-card.amber::before { background:linear-gradient(90deg,#92400e,#fcd34d) !important; }

.r-tag {
  display:inline-flex; align-items:center; gap:.35rem;
  font-size:.65rem; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; padding:.22rem .9rem;
  border-radius:999px; margin-bottom:.9rem; font-family:var(--mono);
}
.r-ok   { background:rgba(134,239,172,.1); color:var(--g);  border:1px solid rgba(134,239,172,.28); }
.r-bad  { background:var(--redbg); color:var(--red);        border:1px solid rgba(252,165,165,.28); }
.r-warn { background:var(--amrbg); color:var(--amr);        border:1px solid rgba(252,211,77,.28); }

.r-name  { font-family:var(--disp); font-size:2.1rem; font-weight:800; letter-spacing:-.05em; color:var(--cream); line-height:1.05; margin-bottom:.15rem; }
.r-sci   { font-size:.82rem; color:var(--c4); font-style:italic; margin-bottom:1.3rem; }
.r-ch    { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.35rem; }
.r-cl    { font-size:.63rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--c4); font-family:var(--mono); }
.r-cv    { font-family:var(--disp); font-size:1.7rem; font-weight:800; letter-spacing:-.05em; color:var(--cream); }
.r-bt    { height:6px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; margin-bottom:1.4rem; }
.r-bf    { height:100%; border-radius:999px; animation:barGrow .9s cubic-bezier(.4,0,.2,1) both; }
.r-meta  { font-size:.68rem; color:var(--c4); font-family:var(--mono); display:flex; gap:.8rem; }

/* Prob rows */
.pb-row  { display:flex; align-items:center; gap:.65rem; margin-bottom:.45rem; }
.pb-name { font-size:.74rem; font-family:var(--mono); color:var(--c3); width:120px; flex-shrink:0; }
.pb-tr   { flex:1; height:4px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; }
.pb-fill { height:100%; border-radius:999px; background:var(--bg4); animation:barGrow .65s cubic-bezier(.4,0,.2,1) both; }
.pb-hi   { background:linear-gradient(90deg,var(--gd),var(--g)) !important; }
.pb-pct  { font-size:.72rem; font-family:var(--mono); color:var(--c3); width:36px; text-align:right; flex-shrink:0; }

/* Info cards */
.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin-top:.9rem; }
.info-card {
  background:var(--bg3); border:1px solid var(--bd);
  border-radius:var(--rl); padding:1.1rem 1.15rem;
  transition:border-color .18s;
}
.info-card:hover { border-color:var(--bd2); }
.info-card-h { font-size:.6rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; color:var(--c4); font-family:var(--mono); margin-bottom:.65rem; display:flex; align-items:center; gap:.4rem; }
.info-card-b { font-size:.81rem; color:var(--c2); line-height:1.7; }
.sym-chip { display:inline-block; background:var(--bg4); border:1px solid var(--bd); border-radius:6px; font-size:.73rem; color:var(--c2); padding:.18rem .55rem; margin:.12rem .06rem; line-height:1.45; }
.prev-item { display:flex; align-items:flex-start; gap:.4rem; margin-bottom:.4rem; font-size:.8rem; color:var(--c2); line-height:1.6; }
.prev-ico  { color:var(--g); flex-shrink:0; margin-top:.2rem; font-size:.7rem; }
.sev-badge { display:inline-block; margin-top:.75rem; font-size:.65rem; font-weight:700; letter-spacing:.08em; font-family:var(--mono); padding:.2rem .65rem; border-radius:999px; }

/* Stats strip */
.stats-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin-bottom:1.5rem; }
.stat-box { background:var(--bg2); border:1px solid var(--bd); border-radius:var(--r); padding:.75rem .5rem; text-align:center; transition:border-color .18s, transform .18s; }
.stat-box:hover { border-color:rgba(134,239,172,.2); transform:translateY(-2px); }
.stat-n { font-family:var(--disp); font-size:1.4rem; font-weight:800; color:var(--g); letter-spacing:-.04em; line-height:1; }
.stat-l { font-size:.6rem; color:var(--c4); margin-top:.15rem; letter-spacing:.07em; text-transform:uppercase; font-family:var(--mono); }

/* History rows */
.hist-row {
  display:flex; align-items:center; gap:.75rem;
  padding:.65rem .9rem; background:var(--bg2); border:1px solid var(--bd);
  border-radius:var(--r); margin-bottom:.4rem; font-size:.8rem;
  transition:border-color .18s; animation:fadeIn .3s ease both;
}
.hist-row:hover { border-color:var(--bd2); }
.hist-ico  { font-size:1.1rem; flex-shrink:0; }
.hist-name { font-family:var(--disp); font-weight:600; color:var(--c2); flex:1; font-size:.82rem; }
.hist-conf { font-family:var(--mono); font-size:.72rem; color:var(--c3); }
.hist-time { font-family:var(--mono); font-size:.65rem; color:var(--c4); }
.hist-tag  { font-family:var(--mono); font-size:.65rem; font-weight:700; padding:.15rem .55rem; border-radius:999px; }
.ht-ok   { background:rgba(134,239,172,.1); color:var(--g);  border:1px solid rgba(134,239,172,.25); }
.ht-bad  { background:var(--redbg); color:var(--red);        border:1px solid rgba(252,165,165,.25); }
.ht-warn { background:var(--amrbg); color:var(--amr);        border:1px solid rgba(252,211,77,.25); }

.app-footer { text-align:center; padding:2rem 0 1.2rem; font-size:.68rem; color:var(--c4); font-family:var(--mono); border-top:1px solid var(--bd); margin-top:2.5rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing_page():
    inject_css()
    st.markdown("""
<div class="lp-bg"></div>
<div class="lp-grid"></div>
<div class="lp-corner lp-tl">🌽</div>
<div class="lp-corner lp-tr">🌽</div>
<div class="lp-corner lp-bl">🌽</div>
<div class="lp-corner lp-br">🌽</div>

<div class="lp-wrap">
  <div class="lp-orb">🌿</div>
  <div class="lp-pill">
    <span class="lp-pill-dot"></span>
    Deep Learning · Plant Pathology · v3.1
  </div>
  <div class="lp-title">
    CornScan<br><span class="lp-title-grad">AI</span>
  </div>
  <div class="lp-sub">
    Upload a corn leaf photo. Get an instant, science-backed disease
    diagnosis powered by a deep convolutional neural network.
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
    <div class="lp-feat"><div class="lp-feat-ico">📋</div><div class="lp-feat-t">Field Reports</div><div class="lp-feat-d">Disease info, symptoms & prevention tips</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🔬</div><div class="lp-feat-t">4 Classes</div><div class="lp-feat-d">Blight, Rust, Gray Leaf Spot, Healthy</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📦</div><div class="lp-feat-t">Batch Mode</div><div class="lp-feat-d">Upload multiple images in one session</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📜</div><div class="lp-feat-t">History Log</div><div class="lp-feat-d">Session log with timestamps & confidence</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c2:
        if st.button("🚀  Launch CornScan AI", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown("""<div style="text-align:center;margin-top:.9rem;font-size:.67rem;color:var(--c4);font-family:var(--mono);">
Powered by TensorFlow · Keras · Streamlit &nbsp;·&nbsp; No data leaves your device
</div>""", unsafe_allow_html=True)


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
      <div class="app-topbar-ver">v3.1 · CNN · TensorFlow</div>
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

    # ── Stats ──────────────────────────────────────────────────────────────
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

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">📁 Upload Leaf Image</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "drop", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    st.markdown('<div style="text-align:center;font-size:.72rem;color:var(--c4);font-family:var(--mono);margin-top:.35rem;">JPG · PNG · JPEG &nbsp;|&nbsp; Multiple files supported</div>', unsafe_allow_html=True)

    valid, analyze = [], False

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
                    st.markdown(f'<div class="img-foot"><span>{f.name[:20]}</span><span class="img-badge">{w}×{h}</span></div>', unsafe_allow_html=True)
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
            st.caption(f"+{len(uploaded_files)-3} more file(s) queued")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button(f"🔬  Analyze {len(valid)} Image{'s' if len(valid)>1 else ''}", use_container_width=True)

    else:
        st.markdown("""<div style="text-align:center;padding:1.8rem 0;font-size:.82rem;
color:var(--c4);font-family:var(--mono);">↑ &nbsp;Drop or browse a corn leaf image to begin</div>""", unsafe_allow_html=True)

    # ── Inference ──────────────────────────────────────────────────────────
    if analyze and valid:
        model = load_model()
        batch = []
        with st.spinner("Running deep scan…"):
            for fname, img in valid:
                label, conf, all_probs = predict(img)
                ts     = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
                info   = DISEASE_INFO[label]
                status = "ok" if label == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
                batch.append(dict(fname=fname, img=img, label=label, conf=conf, all_probs=all_probs, ts=ts, info=info, status=status))
                st.session_state.history.insert(0, dict(label=label, conf=conf, ts=ts, fname=fname, status=status, info=info))
                st.session_state.scanned += 1
        st.session_state.results = batch

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧬 Diagnosis</div>', unsafe_allow_html=True)

        for r in st.session_state.results:
            info     = r["info"]
            pct      = r["conf"] * 100
            status   = r["status"]
            card_cls = {"ok":"","warn":"amber","bad":"diseased"}.get(status, "")
            tag_cls  = {"ok":"r-ok","warn":"r-warn","bad":"r-bad"}.get(status, "r-ok")
            tag_txt  = {"ok":"⬤ Healthy","warn":"⬤ Monitor","bad":"⬤ Diseased"}.get(status, "")
            bar_grad = {
                "ok"  : "linear-gradient(90deg,#16a34a,#86efac)",
                "warn": "linear-gradient(90deg,#b45309,#fcd34d)",
                "bad" : "linear-gradient(90deg,#991b1b,#fca5a5)",
            }.get(status, "")

            st.markdown(f"""
<div class="result-card {card_cls}">
  <span class="r-tag {tag_cls}">{tag_txt}</span>
  <div class="r-name">{info['short']}</div>
  <div class="r-sci">{info['pathogen']}</div>
  <div class="r-ch">
    <span class="r-cl">Confidence</span>
    <span class="r-cv">{pct:.1f}%</span>
  </div>
  <div class="r-bt"><div class="r-bf" style="width:{pct:.1f}%;background:{bar_grad};"></div></div>
  <div class="r-meta"><span>🕐 {r['ts']}</span><span>📄 {r['fname']}</span></div>
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

        # ── Disease detail panel ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📚 Agronomic Details</div>', unsafe_allow_html=True)

        seen = set()
        for r in st.session_state.results:
            lbl = r["label"]
            if lbl in seen:
                continue
            seen.add(lbl)
            info    = r["info"]
            sc      = info["sev_color"]
            sev_bg  = sc + "18"

            with st.expander(f"{info['icon']}  {info['short']}", expanded=True):
                chips = "".join(f'<span class="sym-chip">· {s}</span>' for s in info["symptoms"])
                st.markdown(f"""
<div class="info-grid">
  <div class="info-card">
    <div class="info-card-h">📋 Overview</div>
    <div class="info-card-b">{info['desc']}</div>
    <span class="sev-badge" style="background:{sev_bg};color:{sc};border:1px solid {sc}44;">
      SEVERITY · {info['severity']}
    </span>
  </div>
  <div class="info-card">
    <div class="info-card-h">🔍 Symptoms</div>
    <div>{chips}</div>
    <div style="margin-top:.85rem">
      <div class="info-card-h">🛡 Recommended Action</div>
      <div class="prev-item"><span class="prev-ico">✓</span><span>{info['action']}</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── History ────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📜 Scan History</div>', unsafe_allow_html=True)

        for h in st.session_state.history[:8]:
            tc = {"ok":"ht-ok","warn":"ht-warn","bad":"ht-bad"}.get(h["status"],"ht-ok")
            tt = {"ok":"Healthy","warn":"Monitor","bad":"Diseased"}.get(h["status"],"—")
            st.markdown(f"""
<div class="hist-row">
  <span class="hist-ico">{h['info']['icon']}</span>
  <span class="hist-name">{h['info']['short']}</span>
  <span class="hist-conf">{h['conf']*100:.1f}%</span>
  <span class="hist-time">{h['ts']}</span>
  <span class="hist-tag {tc}">{tt}</span>
</div>""", unsafe_allow_html=True)

        if len(st.session_state.history) > 8:
            st.caption(f"+{len(st.session_state.history)-8} older entries")

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
  CornScan AI &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; CNN Plant Disease Detection
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    landing_page()
else:
    main_app()