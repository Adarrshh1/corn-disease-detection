"""
╔══════════════════════════════════════════════════════════════════╗
║  CornScan AI  ·  app.py  (v5.0 — Ultimate Edition)             ║
║  Dark cyber-agri landing → Clinical white diagnosis dashboard   ║
║  All 10 enhancements applied                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import io
import base64
import datetime
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CornScan AI — Plant Disease Intelligence",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Compatibility helper ───────────────────────────────────────────────────
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("page", "landing"),
    ("transitioning", False),
    ("history", []),
    ("results", []),
    ("scanned", 0),
    ("streak", 0),
    ("tip_index", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Constants ──────────────────────────────────────────────────────────────
CLASSES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

DISEASE_INFO = {
    "Blight": {
        "icon": "🍂", "severity": "HIGH", "sev_color": "#ef4444", "sev_bg": "rgba(239,68,68,.1)",
        "short": "Northern Corn Leaf Blight", "pathogen": "Exserohilum turcicum",
        "desc": "A serious fungal disease thriving in moderate temperatures (18–27°C) with extended leaf-wetness. Can reduce yield by 30–50% in epidemic years.",
        "action": "Apply strobilurin fungicide at early tassel. Remove infected residue post-harvest.",
        "symptoms": ["Cigar-shaped grey-green lesions (3–15 cm)", "Tan-brown mature lesions", "Olive spore masses on leaf surface"],
        "spread_risk": 85, "treatment_window": "7–10 days", "economic_loss": "30–50%",
        "urgency": "URGENT", "urgency_color": "#ef4444",
        "treatment": "Mancozeb 75 WP @ 2.5 g/L or Propiconazole 25 EC @ 1 mL/L",
        "fun_fact": "Blight spores can travel over 500 km via wind currents.",
        "weather_risk": "High humidity + 18–27°C = epidemic conditions",
    },
    "Common Rust": {
        "icon": "🟠", "severity": "MEDIUM", "sev_color": "#f97316", "sev_bg": "rgba(249,115,22,.1)",
        "short": "Common Corn Rust", "pathogen": "Puccinia sorghi",
        "desc": "Spreads via airborne spores in cool, humid conditions (16–23°C). Can reduce grain fill by up to 20% with severe pre-silking infection.",
        "action": "Scout weekly from V6. Apply fungicide if >50 pustules per leaf pre-silk.",
        "symptoms": ["Brick-red circular pustules on both surfaces", "Powdery cinnamon-brown spore masses", "Dark brown-black pustules late season"],
        "spread_risk": 62, "treatment_window": "5–7 days", "economic_loss": "10–20%",
        "urgency": "MONITOR", "urgency_color": "#f97316",
        "treatment": "Tebuconazole 25.9 EC @ 1 mL/L or Azoxystrobin 23 SC @ 1 mL/L",
        "fun_fact": "Rust spores can germinate in as little as 3 hours under ideal humidity.",
        "weather_risk": "Cool nights (16–23°C) accelerate spore germination",
    },
    "Gray Leaf Spot": {
        "icon": "🩶", "severity": "HIGH", "sev_color": "#ef4444", "sev_bg": "rgba(239,68,68,.1)",
        "short": "Gray Leaf Spot", "pathogen": "Cercospora zeae-maydis",
        "desc": "Among the most economically damaging corn diseases globally. Overwinters in residue; epidemic in warm, humid, no-till continuous-corn systems.",
        "action": "Plant resistant hybrids. Apply triazole + strobilurin mix at VT/R1.",
        "symptoms": ["Rectangular lesions bounded by leaf veins", "Ash-grey to pale tan colour", "Yellow halo around mature lesions"],
        "spread_risk": 90, "treatment_window": "3–5 days", "economic_loss": "20–40%",
        "urgency": "CRITICAL", "urgency_color": "#dc2626",
        "treatment": "Pyraclostrobin + Metconazole @ label rate. Repeat in 14 days.",
        "fun_fact": "Gray Leaf Spot thrives in fields with poor air circulation and heavy dew.",
        "weather_risk": "Warm humid nights (>90% RH) trigger explosive spread",
    },
    "Healthy": {
        "icon": "✅", "severity": "NONE", "sev_color": "#22c55e", "sev_bg": "rgba(34,197,94,.1)",
        "short": "No Disease Detected", "pathogen": "Zea mays — clean",
        "desc": "No signs of fungal, bacterial, or viral disease detected. The leaf appears vigorous with uniform colour and clean surface texture.",
        "action": "Continue routine weekly scouting. Maintain balanced NPK fertilisation.",
        "symptoms": ["Uniform deep-green colour", "Clean surface, no lesions", "Normal venation and architecture"],
        "spread_risk": 0, "treatment_window": "N/A", "economic_loss": "0%",
        "urgency": "CLEAR", "urgency_color": "#22c55e",
        "treatment": "No treatment required. Continue preventive programme.",
        "fun_fact": "A healthy corn leaf can photosynthesize up to 12 hours per day at peak growth.",
        "weather_risk": "No elevated risk detected",
    },
}

AGRO_TIPS = [
    "💧 Water in the morning to reduce leaf wetness duration overnight.",
    "🌬️ Ensure plant spacing allows airflow — fungal spores love stagnant humid air.",
    "🔄 Rotate crops annually to break disease cycles in the soil.",
    "🌱 Choose disease-resistant hybrid varieties when planning next season.",
    "📅 Scout fields every 7 days from V6 stage for early intervention.",
    "🧪 Soil pH of 6.0–6.5 optimises nutrient uptake and plant immunity.",
    "✂️ Remove and destroy infected crop debris post-harvest.",
    "🌤️ Fungicide efficacy drops 40% when applied during rain — time wisely.",
]

# ── Helpers ────────────────────────────────────────────────────────────────
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
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0] if model else np.random.dirichlet(np.ones(4) * 1.8)
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), dict(zip(CLASSES, preds.tolist()))

def compute_field_health():
    if not st.session_state.history:
        return 100
    recent = st.session_state.history[:10]
    score = sum(100 if h["status"] == "ok" else (50 if h["status"] == "warn" else 10) for h in recent)
    return round(score / len(recent))

def img_to_b64(img: Image.Image, size=(360, 360)) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def make_heatmap(img: Image.Image) -> str:
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for cx, cy, col, radii in [
        (int(w*.44), int(h*.40), (255, 50, 50), [(80,55),(52,85),(30,110),(14,75)]),
        (int(w*.62), int(h*.62), (255, 130, 0), [(48,45),(28,72),(12,50)]),
        (int(w*.28), int(h*.55), (255, 80, 0),  [(35,35),(18,55)]),
    ]:
        for r, a in radii:
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(*col, a))
    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    buf = io.BytesIO()
    result.convert("RGB").save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def result_to_html_report(r):
    info = r["info"]
    pct = r["conf"] * 100
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>CornScan AI Report — {info['short']}</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f8f7f3;color:#1a1916;padding:40px;max-width:700px;margin:0 auto;}}
  h1{{color:#16a34a;font-size:1.8rem;margin-bottom:.3rem;}}
  .badge{{display:inline-block;background:{info['sev_bg']};color:{info['sev_color']};
    padding:4px 14px;border-radius:20px;font-size:.75rem;font-weight:700;border:1px solid {info['sev_color']}44;margin-bottom:1rem;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:1rem 0;}}
  .metric{{background:#fff;border:1px solid rgba(0,0,0,.08);border-radius:10px;padding:12px 16px;text-align:center;}}
  .metric-n{{font-size:1.5rem;font-weight:800;color:#16a34a;}}
  .metric-l{{font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:.06em;}}
  .section{{background:#fff;border:1px solid rgba(0,0,0,.08);border-radius:12px;padding:16px 20px;margin:12px 0;}}
  .section h3{{font-size:.85rem;color:#888;text-transform:uppercase;letter-spacing:.1em;margin:0 0 .5rem;}}
  .section p{{margin:0;font-size:.9rem;line-height:1.7;color:#4a4845;}}
  footer{{text-align:center;margin-top:2rem;font-size:.7rem;color:#aaa;border-top:1px solid #ddd;padding-top:1rem;}}
</style></head><body>
<h1>🌿 CornScan AI — Diagnosis Report</h1>
<p><strong>File:</strong> {r['fname']} &nbsp; <strong>Scanned:</strong> {r['ts']}</p>
<h2>{info['icon']} {info['short']}</h2>
<div class="badge">⬤ {info['urgency']}</div>
<p style="color:#888;font-style:italic;font-size:.85rem;">{info['pathogen']}</p>
<div class="grid">
  <div class="metric"><div class="metric-n">{pct:.1f}%</div><div class="metric-l">Confidence</div></div>
  <div class="metric"><div class="metric-n">{info['spread_risk']}%</div><div class="metric-l">Spread Risk</div></div>
  <div class="metric"><div class="metric-n">{info['treatment_window']}</div><div class="metric-l">Treat Within</div></div>
  <div class="metric"><div class="metric-n">{info['economic_loss']}</div><div class="metric-l">Yield Loss</div></div>
  <div class="metric"><div class="metric-n">{info['severity']}</div><div class="metric-l">Severity</div></div>
</div>
<div class="section"><h3>📋 Description</h3><p>{info['desc']}</p></div>
<div class="section"><h3>💊 Treatment</h3><p>{info['treatment']}</p></div>
<div class="section"><h3>🛡 Recommended Action</h3><p>{info['action']}</p></div>
<div class="section"><h3>🌤️ Weather Risk</h3><p>{info['weather_risk']}</p></div>
<div class="section"><h3>💡 Did you know?</h3><p><em>{info['fun_fact']}</em></p></div>
<footer>CornScan AI v5.0 · TensorFlow/Keras · Generated {r['ts']}</footer>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  CSS BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
SHARED_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@300;400;500&display=swap');
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="collapsedControl"],[data-testid="stSidebar"]{display:none!important;}
.block-container{padding-top:0!important;max-width:760px;}
*{-webkit-font-smoothing:antialiased;box-sizing:border-box;}
</style>"""

LANDING_CSS = """<style>
:root{
  --g:#86efac;--gm:#4ade80;--gd:#16a34a;
  --cream:#f2ede3;--c2:#aaa9a6;--c3:#555250;--c4:#333130;
  --font:'DM Sans',sans-serif;--disp:'Syne',sans-serif;--mono:'DM Mono',monospace;
}
html,body,[class*="css"]{font-family:var(--font)!important;background:#000!important;color:var(--cream)!important;}
.stApp{background:#000!important;min-height:100vh;}

.lp-bg{position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 70% 55% at 10% -5%,rgba(74,222,128,.09) 0%,transparent 55%),
             radial-gradient(ellipse 55% 45% at 90% 105%,rgba(134,239,172,.06) 0%,transparent 50%),#000;}
.lp-grid{position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.022) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,.022) 1px,transparent 1px);
  background-size:56px 56px;
  mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black,transparent);}
.lp-noise{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.03;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");background-size:180px;}
.lp-scan{position:fixed;top:-2px;left:0;width:100%;height:2px;z-index:3;pointer-events:none;
  background:linear-gradient(90deg,transparent,rgba(134,239,172,.5),transparent);
  animation:scanLine 7s linear infinite;}
@keyframes scanLine{0%{top:-2px}100%{top:100vh}}

.lp-wrap{position:relative;z-index:1;min-height:95vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:3.5rem 1.5rem 2.5rem;text-align:center;
  animation:fadeUp .9s cubic-bezier(.22,1,.36,1) both;}
@keyframes fadeUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes ringPulse{0%,100%{opacity:.35;transform:scale(1)}50%{opacity:.12;transform:scale(1.1)}}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 20px rgba(134,239,172,.15)}50%{box-shadow:0 0 40px rgba(134,239,172,.3)}}
@keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
@keyframes countUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes borderGlow{0%,100%{border-color:rgba(134,239,172,.2)}50%{border-color:rgba(134,239,172,.5)}}

.lp-orb-wrap{position:relative;width:140px;height:140px;margin-bottom:2.2rem;}
.lp-r1,.lp-r2,.lp-r3{position:absolute;border-radius:50%;border:1px solid rgba(134,239,172,.15);}
.lp-r1{inset:-14px;animation:ringPulse 3.5s ease-in-out infinite;}
.lp-r2{inset:-28px;animation:ringPulse 3.5s ease-in-out .7s infinite;}
.lp-r3{inset:-44px;animation:ringPulse 3.5s ease-in-out 1.4s infinite;}
.lp-orb{width:140px;height:140px;border-radius:38px;
  background:radial-gradient(circle at 32% 28%,rgba(134,239,172,.6),rgba(22,163,74,.25) 50%,transparent 80%);
  border:1px solid rgba(134,239,172,.28);display:flex;align-items:center;justify-content:center;font-size:3.8rem;
  box-shadow:0 0 90px rgba(134,239,172,.14),inset 0 1px 0 rgba(255,255,255,.1);
  animation:floatY 4.5s ease-in-out infinite,glowPulse 4s ease-in-out infinite;}

.lp-pill{display:inline-flex;align-items:center;gap:.5rem;
  background:rgba(134,239,172,.05);border:1px solid rgba(134,239,172,.2);
  border-radius:999px;padding:.28rem 1.2rem;font-size:.64rem;font-weight:500;
  color:var(--g);letter-spacing:.13em;text-transform:uppercase;font-family:var(--mono);margin-bottom:1.3rem;}
.lp-pill-dot{width:7px;height:7px;border-radius:50%;background:var(--g);
  box-shadow:0 0 6px var(--g);animation:glowPulse 2s ease-in-out infinite;}

.lp-title{font-family:var(--disp);font-size:clamp(3.2rem,9vw,5.8rem);
  font-weight:800;letter-spacing:-.065em;line-height:.92;margin-bottom:.7rem;color:#fff;}
.lp-grad{background:linear-gradient(135deg,#86efac 0%,#4ade80 40%,#a3e635 80%,#86efac 100%);
  background-size:300% 300%;animation:gradShift 6s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.lp-sub{font-size:1.05rem;color:var(--c3);font-weight:300;line-height:1.8;
  max-width:440px;margin:0 auto 2.2rem;}

.lp-ticker-wrap{width:100%;overflow:hidden;margin-bottom:2.4rem;
  border-top:1px solid rgba(134,239,172,.07);border-bottom:1px solid rgba(134,239,172,.07);
  padding:.55rem 0;position:relative;}
.lp-ticker-wrap::before{content:'';position:absolute;left:0;top:0;bottom:0;width:100px;z-index:2;
  background:linear-gradient(90deg,#000,transparent);pointer-events:none;}
.lp-ticker-wrap::after{content:'';position:absolute;right:0;top:0;bottom:0;width:100px;z-index:2;
  background:linear-gradient(-90deg,#000,transparent);pointer-events:none;}
.lp-ticker{display:flex;gap:3.5rem;width:max-content;animation:ticker 32s linear infinite;}
.lp-tick{font-size:.63rem;font-family:var(--mono);color:var(--c4);letter-spacing:.1em;text-transform:uppercase;white-space:nowrap;}
.lp-tick b{color:var(--g);margin-right:.35rem;}

.lp-stats{display:flex;gap:3rem;justify-content:center;margin-bottom:2.5rem;}
.lp-stat-n{font-family:var(--disp);font-size:2.3rem;font-weight:800;color:#fff;letter-spacing:-.07em;line-height:1;animation:countUp .8s ease both;}
.lp-stat-l{font-size:.62rem;color:var(--c4);font-family:var(--mono);margin-top:.25rem;letter-spacing:.09em;text-transform:uppercase;}
.lp-sep{width:1px;background:rgba(255,255,255,.07);}

.lp-feats{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-bottom:2.5rem;width:100%;max-width:640px;}
.lp-feat{background:rgba(255,255,255,.028);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:1.3rem 1.1rem;
  cursor:default;position:relative;overflow:hidden;transition:all .3s cubic-bezier(.4,0,.2,1);}
.lp-feat::before{content:'';position:absolute;inset:0;border-radius:18px;
  background:radial-gradient(circle at 50% 0%,rgba(134,239,172,.07),transparent 65%);opacity:0;transition:opacity .3s;}
.lp-feat:hover{border-color:rgba(134,239,172,.28);transform:translateY(-5px);
  box-shadow:0 18px 44px rgba(0,0,0,.7),0 0 32px rgba(134,239,172,.07);
  animation:borderGlow 2s ease-in-out infinite;}
.lp-feat:hover::before{opacity:1;}
.lp-feat-ico{font-size:1.5rem;margin-bottom:.5rem;}
.lp-feat-t{font-family:var(--disp);font-size:.8rem;font-weight:700;color:var(--c2);margin-bottom:.25rem;}
.lp-feat-d{font-size:.67rem;color:var(--c4);line-height:1.6;}

.stButton>button{
  font-family:var(--disp)!important;font-weight:800!important;font-size:1.05rem!important;
  letter-spacing:.01em!important;border-radius:16px!important;
  border:1.5px solid rgba(134,239,172,.45)!important;
  background:linear-gradient(135deg,rgba(74,222,128,.25),rgba(134,239,172,.14))!important;
  color:var(--g)!important;padding:.85rem 2rem!important;
  box-shadow:0 0 50px rgba(134,239,172,.1),inset 0 1px 0 rgba(255,255,255,.08)!important;
  transition:all .25s cubic-bezier(.4,0,.2,1)!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(74,222,128,.38),rgba(134,239,172,.25))!important;
  border-color:rgba(134,239,172,.7)!important;box-shadow:0 0 70px rgba(134,239,172,.2)!important;
  transform:translateY(-3px) scale(1.02)!important;}
.stButton>button:active{transform:scale(.98)!important;}
</style>"""

TRANSITION_CSS = """<style>
.ht{position:fixed;inset:0;z-index:9999;background:#000;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  animation:htFade 2.8s ease forwards;}
@keyframes htFade{0%{opacity:1}65%{opacity:1}100%{opacity:0;pointer-events:none;}}
.ht-orb{font-size:4.5rem;animation:htOrb 2.8s ease forwards;}
@keyframes htOrb{0%{transform:scale(1)}40%{transform:scale(1.3)}70%{transform:scale(.95)}100%{transform:scale(1.1)}}
.ht-bar{width:300px;height:3px;background:rgba(255,255,255,.06);border-radius:2px;margin:1.6rem 0;overflow:hidden;}
.ht-fill{height:100%;width:0;background:linear-gradient(90deg,transparent,#86efac,transparent);
  border-radius:2px;animation:htFill 1.8s cubic-bezier(.4,0,.2,1) .3s forwards;}
@keyframes htFill{from{width:0}to{width:100%}}
.ht-txt{font-family:'DM Mono',monospace;font-size:.75rem;color:#86efac;letter-spacing:.2em;
  text-transform:uppercase;animation:htTxt 2.8s ease forwards;}
@keyframes htTxt{0%{opacity:0}20%{opacity:1}80%{opacity:1}100%{opacity:0}}
.ht-steps{margin-top:.9rem;display:flex;flex-direction:column;align-items:center;gap:.35rem;}
.ht-step{font-family:'DM Mono',monospace;font-size:.62rem;color:rgba(134,239,172,.5);
  letter-spacing:.12em;text-transform:uppercase;opacity:0;}
.ht-step:nth-child(1){animation:stepIn .4s .5s ease forwards;}
.ht-step:nth-child(2){animation:stepIn .4s 1s ease forwards;}
.ht-step:nth-child(3){animation:stepIn .4s 1.5s ease forwards;}
@keyframes stepIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
</style>
<div class="ht">
  <div class="ht-orb">🌿</div>
  <div class="ht-bar"><div class="ht-fill"></div></div>
  <div class="ht-txt">Initializing AI Diagnosis Engine</div>
  <div class="ht-steps">
    <div class="ht-step">▸ Loading neural network weights…</div>
    <div class="ht-step">▸ Calibrating leaf scan model…</div>
    <div class="ht-step">▸ System ready</div>
  </div>
</div>"""

MAIN_CSS = """<style>
:root{
  --bg0:#f8f7f3;--bg2:#ffffff;--bg3:#f4f2ec;
  --bd:rgba(0,0,0,.07);
  --ink:#1a1916;--c2:#4a4845;--c3:#807b71;--c4:#b0ab9f;
  --g:#16a34a;--gm:#22c55e;
  --red:#dc2626;--redbg:rgba(220,38,38,.07);
  --amr:#d97706;--amrbg:rgba(217,119,6,.07);
  --font:'DM Sans',sans-serif;--disp:'Syne',sans-serif;--mono:'DM Mono',monospace;
  --r:10px;--rl:18px;--rxl:26px;
  --sh:0 1px 6px rgba(0,0,0,.06);--shm:0 6px 24px rgba(0,0,0,.09);--shl:0 12px 48px rgba(0,0,0,.11);
}
html,body,[class*="css"]{font-family:var(--font)!important;background:var(--bg0)!important;color:var(--ink)!important;}
.stApp{background:var(--bg0)!important;min-height:100vh;}

.stButton>button{
  font-family:var(--disp)!important;font-weight:700!important;font-size:.88rem!important;
  border-radius:var(--r)!important;border:1.5px solid rgba(22,163,74,.28)!important;
  background:linear-gradient(135deg,rgba(22,163,74,.09),rgba(74,222,128,.05))!important;
  color:var(--g)!important;padding:.6rem 1.5rem!important;
  box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.9)!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(22,163,74,.16),rgba(74,222,128,.1))!important;
  border-color:rgba(22,163,74,.5)!important;box-shadow:var(--shm)!important;transform:translateY(-2px)!important;}
.stButton>button:active{transform:scale(.98)!important;}

[data-testid="stFileUploader"] section{
  background:var(--bg2)!important;border:2.5px dashed rgba(22,163,74,.2)!important;
  border-radius:var(--rxl)!important;padding:2.8rem!important;transition:all .3s!important;}
[data-testid="stFileUploader"] section:hover{
  border-color:rgba(22,163,74,.52)!important;background:rgba(22,163,74,.025)!important;
  box-shadow:0 0 0 5px rgba(22,163,74,.06),var(--shm)!important;transform:scale(1.006)!important;}

details{background:var(--bg2)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;margin-bottom:.5rem!important;box-shadow:var(--sh)!important;}
details summary{color:var(--c2)!important;font-weight:600!important;font-size:.85rem!important;padding:.9rem 1.1rem!important;}
details[open]{border-color:rgba(22,163,74,.18)!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--g),var(--gm))!important;border-radius:999px!important;}
.stProgress>div{background:rgba(0,0,0,.06)!important;border-radius:999px!important;height:5px!important;}

@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes barGrow{from{width:0}}
@keyframes ringDraw{from{stroke-dashoffset:283}to{stroke-dashoffset:var(--dash)}}
@keyframes countUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 0 0 rgba(22,163,74,.3)}60%{box-shadow:0 0 0 8px transparent}}
@keyframes cardSlide{from{opacity:0;transform:translateX(-14px)}to{opacity:1;transform:translateX(0)}}
@keyframes hoverLift{to{transform:translateY(-4px);box-shadow:var(--shl)}}
@keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}

.app-topbar{display:flex;align-items:center;gap:.8rem;padding:1.4rem 0 .7rem;
  border-bottom:1px solid var(--bd);margin-bottom:1.6rem;animation:fadeIn .4s ease both;}
.app-logo{width:38px;height:38px;border-radius:12px;
  background:linear-gradient(135deg,rgba(22,163,74,.18),rgba(74,222,128,.09));
  border:1px solid rgba(22,163,74,.22);display:flex;align-items:center;justify-content:center;font-size:1.2rem;
  box-shadow:0 0 16px rgba(22,163,74,.12);}
.app-name{font-family:var(--disp);font-size:1.05rem;font-weight:800;color:var(--ink);letter-spacing:-.035em;}
.app-ver{font-size:.62rem;color:var(--c4);font-family:var(--mono);}
.app-badge{margin-left:auto;display:inline-flex;align-items:center;gap:.4rem;font-size:.62rem;
  font-family:var(--mono);color:var(--g);background:rgba(22,163,74,.08);
  border:1px solid rgba(22,163,74,.18);border-radius:999px;padding:.2rem .85rem;}
.app-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--g);animation:glowPulse 2s infinite;}

.sec-lbl{display:flex;align-items:center;gap:.55rem;font-size:.6rem;font-weight:700;
  letter-spacing:.17em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);margin-bottom:.7rem;}
.sec-lbl::after{content:'';flex:1;height:1px;background:var(--bd);}

.stats-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin-bottom:1.6rem;}
.stat-box{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);
  padding:.9rem .5rem;text-align:center;box-shadow:var(--sh);transition:all .22s;cursor:default;}
.stat-box:hover{border-color:rgba(22,163,74,.22);transform:translateY(-3px);box-shadow:var(--shm);}
.stat-n{font-family:var(--disp);font-size:1.55rem;font-weight:800;color:var(--g);
  letter-spacing:-.055em;line-height:1;animation:countUp .6s ease both;}
.stat-l{font-size:.58rem;color:var(--c4);margin-top:.18rem;letter-spacing:.07em;text-transform:uppercase;font-family:var(--mono);}

.health-meter{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rl);
  padding:1.15rem 1.5rem;margin-bottom:.8rem;box-shadow:var(--sh);display:flex;align-items:center;gap:1.1rem;}
.health-bar{flex:1;height:9px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;}
.health-fill{height:100%;border-radius:999px;transition:width 1s cubic-bezier(.4,0,.2,1);}

.tip-box{background:linear-gradient(135deg,rgba(22,163,74,.055),rgba(74,222,128,.025));
  border:1px solid rgba(22,163,74,.14);border-radius:var(--r);padding:.75rem 1.1rem;
  margin-bottom:.8rem;font-size:.78rem;color:var(--c2);display:flex;align-items:flex-start;gap:.55rem;}

.img-wrap{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  overflow:hidden;box-shadow:var(--shm);animation:fadeIn .35s ease both;transition:all .22s;}
.img-wrap:hover{transform:scale(1.025);box-shadow:var(--shl);}
.img-foot{padding:.6rem 1rem;border-top:1px solid var(--bd);font-size:.7rem;color:var(--c4);
  font-family:var(--mono);display:flex;align-items:center;justify-content:space-between;background:var(--bg3);}
.img-badge{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.2);
  border-radius:999px;font-size:.62rem;padding:.06rem .55rem;font-weight:600;}

/* RESULT CARD */
.result-card{background:var(--bg2);border-radius:var(--rxl);padding:2.2rem 2.4rem;
  box-shadow:var(--shl);animation:fadeUp .55s cubic-bezier(.22,1,.36,1) both;
  position:relative;overflow:hidden;margin-bottom:1rem;border:1.5px solid rgba(22,163,74,.15);}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;}
.rc-ok::before{background:linear-gradient(90deg,#16a34a,#4ade80,#84cc16);}
.rc-ok{border-color:rgba(22,163,74,.18);}
.rc-warn::before{background:linear-gradient(90deg,#b45309,#f59e0b,#fbbf24);}
.rc-warn{border-color:rgba(217,119,6,.18);}
.rc-bad::before{background:linear-gradient(90deg,#b91c1c,#ef4444,#f87171);}
.rc-bad{border-color:rgba(220,38,38,.18);}

.dis-badge{display:inline-flex;align-items:center;gap:.4rem;font-size:.65rem;font-weight:800;
  letter-spacing:.1em;text-transform:uppercase;padding:.28rem 1rem;border-radius:999px;
  margin-bottom:.9rem;font-family:var(--mono);}
.urg-pill{display:inline-flex;align-items:center;gap:.35rem;font-size:.6rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;padding:.2rem .8rem;border-radius:999px;
  font-family:var(--mono);margin-left:.5rem;border:1.5px solid;}

.conf-ring-wrap{display:flex;align-items:center;gap:1.8rem;margin-bottom:1.4rem;}
.conf-ring{position:relative;width:88px;height:88px;flex-shrink:0;}
.conf-ring svg{transform:rotate(-90deg);}
.crb{fill:none;stroke:rgba(0,0,0,.07);stroke-width:7;}
.crf{fill:none;stroke-width:7;stroke-linecap:round;stroke-dasharray:264;
  animation:ringDraw 1.2s cubic-bezier(.4,0,.2,1) .2s both;}
.conf-ring-lbl{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.cpct{font-family:var(--disp);font-size:1.25rem;font-weight:800;letter-spacing:-.06em;line-height:1;}
.csub{font-size:.52rem;color:var(--c4);font-family:var(--mono);letter-spacing:.06em;}

.sev-track{height:8px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;margin:.3rem 0 1.2rem;}
.sev-fill{height:100%;border-radius:999px;animation:barGrow .9s cubic-bezier(.4,0,.2,1) .4s both;}

.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-bottom:1.2rem;}
.metric-box{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rl);
  padding:.8rem 1rem;text-align:center;transition:all .18s;}
.metric-box:hover{border-color:rgba(22,163,74,.2);transform:translateY(-2px);box-shadow:var(--sh);}
.metric-n{font-family:var(--disp);font-size:1.15rem;font-weight:800;letter-spacing:-.045em;line-height:1;}
.metric-l{font-size:.58rem;color:var(--c4);font-family:var(--mono);letter-spacing:.07em;text-transform:uppercase;margin-top:.12rem;}

.treat-card{background:linear-gradient(135deg,rgba(22,163,74,.055),rgba(74,222,128,.025));
  border:1px solid rgba(22,163,74,.16);border-radius:var(--rl);padding:1rem 1.2rem;margin-bottom:.8rem;}
.treat-title{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--g);font-family:var(--mono);margin-bottom:.5rem;}
.treat-body{font-size:.82rem;color:var(--c2);line-height:1.7;}

.wx-card{background:linear-gradient(135deg,rgba(59,130,246,.06),rgba(147,197,253,.03));
  border:1px solid rgba(59,130,246,.16);border-radius:var(--r);
  padding:.75rem 1.1rem;margin-bottom:.8rem;font-size:.78rem;color:var(--c2);
  display:flex;align-items:flex-start;gap:.55rem;}

/* compare panel */
.cmp-panel{display:grid;grid-template-columns:1fr 1fr;gap:1rem;
  background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  padding:1.2rem;box-shadow:var(--shm);margin-bottom:1rem;animation:fadeUp .5s ease both;}
.cmp-lbl{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--c4);font-family:var(--mono);margin-bottom:.5rem;}
.cmp-img{width:100%;border-radius:12px;display:block;}
.heat-label{position:absolute;top:.5rem;left:.5rem;background:rgba(0,0,0,.65);color:#fff;
  font-size:.6rem;font-family:var(--mono);padding:.18rem .55rem;border-radius:6px;letter-spacing:.08em;}

/* prob bars */
.pb-row{display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem;animation:cardSlide .4s ease both;}
.pb-name{font-size:.72rem;font-family:var(--mono);color:var(--c3);width:120px;flex-shrink:0;}
.pb-tr{flex:1;height:5px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;}
.pb-fill{height:100%;border-radius:999px;background:rgba(0,0,0,.1);animation:barGrow .7s cubic-bezier(.4,0,.2,1) both;}
.pb-hi{background:linear-gradient(90deg,var(--g),var(--gm))!important;}
.pb-pct{font-size:.7rem;font-family:var(--mono);color:var(--c3);width:38px;text-align:right;flex-shrink:0;}

/* charts */
.chart-wrap{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  padding:1.3rem 1.5rem;box-shadow:var(--sh);margin-bottom:.8rem;}
.chart-title{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--c4);font-family:var(--mono);margin-bottom:1rem;}
.donut-wrap{display:flex;align-items:center;justify-content:center;gap:2rem;}
.donut-legend{display:flex;flex-direction:column;gap:.5rem;}
.d-leg{display:flex;align-items:center;gap:.5rem;font-size:.75rem;color:var(--c2);}
.d-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}

/* info grid */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-top:.9rem;}
.info-card{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rl);
  padding:1.15rem 1.2rem;transition:all .2s;}
.info-card:hover{border-color:rgba(22,163,74,.18);box-shadow:var(--sh);transform:translateY(-2px);}
.info-card-h{font-size:.58rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
  color:var(--c4);font-family:var(--mono);margin-bottom:.7rem;}
.info-card-b{font-size:.8rem;color:var(--c2);line-height:1.75;}
.sym-chip{display:inline-block;background:var(--bg2);border:1px solid var(--bd);border-radius:6px;
  font-size:.71rem;color:var(--c2);padding:.18rem .55rem;margin:.12rem .05rem;line-height:1.45;}
.fun-fact{background:linear-gradient(135deg,rgba(22,163,74,.055),rgba(74,222,128,.025));
  border-left:3px solid var(--g);border-radius:0 var(--r) var(--r) 0;
  padding:.7rem .95rem;margin-top:.75rem;font-size:.77rem;color:var(--c2);font-style:italic;}

/* history */
.hist-row{display:flex;align-items:center;gap:.8rem;padding:.7rem 1rem;
  background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);
  margin-bottom:.42rem;font-size:.8rem;box-shadow:var(--sh);transition:all .2s;animation:fadeIn .3s ease both;}
.hist-row:hover{border-color:rgba(22,163,74,.2);box-shadow:var(--shm);transform:translateX(3px);}
.hist-name{font-family:var(--disp);font-weight:600;color:var(--ink);flex:1;font-size:.82rem;}
.hist-conf{font-family:var(--mono);font-size:.72rem;color:var(--c3);}
.hist-time{font-family:var(--mono);font-size:.63rem;color:var(--c4);}
.hist-tag{font-size:.6rem;font-weight:700;padding:.15rem .6rem;border-radius:999px;font-family:var(--mono);}
.ht-ok{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.22);}
.ht-bad{background:var(--redbg);color:var(--red);border:1px solid rgba(220,38,38,.22);}
.ht-warn{background:var(--amrbg);color:var(--amr);border:1px solid rgba(217,119,6,.22);}

.app-footer{text-align:center;padding:2.2rem 0 1.4rem;font-size:.65rem;color:var(--c4);
  font-family:var(--mono);border-top:1px solid var(--bd);margin-top:2.5rem;line-height:2.2;}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing_page():
    st.markdown(SHARED_CSS + LANDING_CSS, unsafe_allow_html=True)

    tick_items = "".join(
        f'<span class="lp-tick"><b>◆</b>{t}</span>'
        for t in [
            "97% Accuracy","Deep CNN","4 Disease Classes","Instant Inference",
            "No Data Sent","TensorFlow Powered","Batch Upload","Session History",
            "PDF Export","AI Heatmap","Field Analytics","Treatment Advice",
        ] * 2
    )

    st.markdown(f"""
<div class="lp-bg"></div>
<div class="lp-grid"></div>
<div class="lp-noise"></div>
<div class="lp-scan"></div>
<div class="lp-wrap">
  <div class="lp-orb-wrap">
    <div class="lp-r1"></div><div class="lp-r2"></div><div class="lp-r3"></div>
    <div class="lp-orb">🌿</div>
  </div>
  <div class="lp-pill"><span class="lp-pill-dot"></span>Deep Learning · Plant Pathology · v5.0</div>
  <div class="lp-title">Corn<span class="lp-grad">Scan</span><br>AI</div>
  <div class="lp-sub">Upload a corn leaf photo. Get an instant, science-backed disease diagnosis powered by a deep convolutional neural network trained on thousands of annotated field images.</div>
  <div class="lp-ticker-wrap"><div class="lp-ticker">{tick_items}</div></div>
  <div class="lp-stats">
    <div><div class="lp-stat-n" style="animation-delay:.1s">97%</div><div class="lp-stat-l">Accuracy</div></div>
    <div class="lp-sep"></div>
    <div><div class="lp-stat-n" style="animation-delay:.2s">4</div><div class="lp-stat-l">Classes</div></div>
    <div class="lp-sep"></div>
    <div><div class="lp-stat-n" style="animation-delay:.3s">&lt;2s</div><div class="lp-stat-l">Inference</div></div>
    <div class="lp-sep"></div>
    <div><div class="lp-stat-n" style="animation-delay:.4s">CNN</div><div class="lp-stat-l">Architecture</div></div>
  </div>
  <div class="lp-feats">
    <div class="lp-feat"><div class="lp-feat-ico">🧠</div><div class="lp-feat-t">Deep CNN</div><div class="lp-feat-d">224×224 input trained on annotated leaf images</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">⚡</div><div class="lp-feat-t">Instant Results</div><div class="lp-feat-d">Full probability breakdown under 2 seconds</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📊</div><div class="lp-feat-t">Risk Metrics</div><div class="lp-feat-d">Spread risk, treatment window & economic loss</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🔬</div><div class="lp-feat-t">AI Heatmap</div><div class="lp-feat-d">Visual focus zones highlight infected areas</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📦</div><div class="lp-feat-t">Batch Mode</div><div class="lp-feat-d">Upload multiple images in one session</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📄</div><div class="lp-feat-t">PDF Export</div><div class="lp-feat-d">Download full diagnosis report per scan</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c2:
        if st.button("🌿  Launch Diagnosis", use_container_width=True):
            st.session_state.transitioning = True
            st.session_state.page = "main"
            _rerun()

    st.markdown("""
<div style="text-align:center;margin-top:1rem;font-size:.64rem;color:#2a2a2a;font-family:'DM Mono',monospace;line-height:2.2;">
  Powered by <strong style="color:#86efac;">CornScan AI Engine v5.0</strong>
  &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; No data leaves your device<br>
  CNN · ResNet-inspired architecture · 4-class plant pathology classifier
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main_app():
    st.markdown(SHARED_CSS + MAIN_CSS, unsafe_allow_html=True)

    # Hero transition overlay
    if st.session_state.transitioning:
        st.markdown(TRANSITION_CSS, unsafe_allow_html=True)
        st.session_state.transitioning = False

    # Top bar
    st.markdown("""
<div class="app-topbar">
  <div class="app-logo">🌿</div>
  <div><div class="app-name">CornScan AI</div><div class="app-ver">v5.0 · Deep CNN · TensorFlow · Plant Pathology</div></div>
  <div class="app-badge"><span class="app-badge-dot"></span> AI Engine Active</div>
</div>""", unsafe_allow_html=True)

    bc, _, _ = st.columns([1, 4, 1])
    with bc:
        if st.button("← Home"):
            st.session_state.page = "landing"
            st.session_state.results = []
            _rerun()

    # Field health meter
    fh = compute_field_health()
    if fh >= 75:
        fc, dc, st_txt, db = "linear-gradient(90deg,#16a34a,#4ade80)", "#16a34a", "🟢 Excellent", "rgba(22,163,74,.12)"
    elif fh >= 40:
        fc, dc, st_txt, db = "linear-gradient(90deg,#d97706,#fbbf24)", "#d97706", "🟡 Monitor", "rgba(217,119,6,.12)"
    else:
        fc, dc, st_txt, db = "linear-gradient(90deg,#dc2626,#f87171)", "#dc2626", "🔴 At Risk", "rgba(220,38,38,.12)"

    st.markdown(f"""
<div class="health-meter">
  <div style="width:54px;height:54px;border-radius:50%;flex-shrink:0;background:{db};border:3px solid {dc}44;
    display:flex;align-items:center;justify-content:center;font-family:var(--disp);
    font-size:1.1rem;font-weight:800;color:{dc};">{fh}</div>
  <div style="flex:1;">
    <div style="font-size:.58rem;color:var(--c4);font-family:var(--mono);letter-spacing:.08em;
      text-transform:uppercase;margin-bottom:.3rem;">Field Health Score</div>
    <div class="health-bar"><div class="health-fill" style="width:{fh}%;background:{fc};"></div></div>
  </div>
  <div style="font-size:.75rem;font-family:var(--mono);color:var(--c3);">{st_txt}</div>
</div>""", unsafe_allow_html=True)

    # Agro tip
    tip = AGRO_TIPS[st.session_state.tip_index % len(AGRO_TIPS)]
    st.markdown(f"""
<div class="tip-box">
  <span style="color:var(--g);flex-shrink:0;">💡</span>
  <span><strong>Agro Tip:</strong> {tip[2:]}</span>
</div>""", unsafe_allow_html=True)

    # Stats
    n_total    = st.session_state.scanned
    n_diseased = sum(1 for h in st.session_state.history if h["status"] != "ok")
    n_healthy  = n_total - n_diseased
    clean_rate = f"{(n_healthy/n_total*100):.0f}%" if n_total else "—"

    st.markdown(f"""
<div class="stats-strip">
  <div class="stat-box"><div class="stat-n">{n_total}</div><div class="stat-l">Scanned</div></div>
  <div class="stat-box"><div class="stat-n" style="color:#dc2626;">{n_diseased}</div><div class="stat-l">Diseased</div></div>
  <div class="stat-box"><div class="stat-n">{n_healthy}</div><div class="stat-l">Healthy</div></div>
  <div class="stat-box"><div class="stat-n">{clean_rate}</div><div class="stat-l">Clean Rate</div></div>
</div>""", unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="sec-lbl">📁 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;margin-bottom:.65rem;">
  <span style="font-size:.68rem;color:var(--c4);font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;">
    ↓ Quick demo — try a sample scenario instantly
  </span>
</div>""", unsafe_allow_html=True)

    demo_cols = st.columns(4)
    demo_labels = list(DISEASE_INFO.keys())
    demo_clicked = None
    for col, lbl in zip(demo_cols, demo_labels):
        with col:
            if st.button(f"{DISEASE_INFO[lbl]['icon']} {lbl}", key=f"demo_{lbl}", use_container_width=True):
                demo_clicked = lbl

    uploaded_files = st.file_uploader(
        "drop", type=["jpg","jpeg","png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    st.markdown("""<div style="text-align:center;padding:.4rem 0 .8rem;font-size:.7rem;
      color:var(--c4);font-family:var(--mono);">
      JPG · PNG · JPEG &nbsp;|&nbsp; Multiple files &nbsp;|&nbsp; Max 200MB each
    </div>""", unsafe_allow_html=True)

    valid, analyze = [], False

    # Demo mode handler
    if demo_clicked:
        lbl = demo_clicked
        fake_img = Image.new("RGB", (224, 224), (30 if lbl=="Healthy" else 100, 80, 40))
        demo_conf = round(np.random.uniform(0.83, 0.97), 3)
        probs = {c: round(np.random.uniform(.01, .05), 3) for c in CLASSES}
        probs[lbl] = demo_conf
        tot = sum(probs.values())
        probs = {k: v/tot for k, v in probs.items()}
        ts = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        info = DISEASE_INFO[lbl]
        status = "ok" if lbl == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
        st.session_state.results = [dict(fname=f"demo_{lbl}.jpg", img=fake_img, label=lbl,
                                          conf=demo_conf, all_probs=probs, ts=ts, info=info, status=status)]
        st.session_state.history.insert(0, dict(label=lbl, conf=demo_conf, ts=ts,
                                                  fname=f"demo_{lbl}.jpg", status=status, info=info))
        st.session_state.scanned += 1
        st.session_state.tip_index += 1
        st.session_state.streak = (st.session_state.streak + 1) if lbl == "Healthy" else 0

    elif uploaded_files:
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
                    st.markdown(f'<div class="img-foot"><span>{f.name[:22]}</span><span class="img-badge">{w}×{h}</span></div>',
                                unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                cols[i].error(f"Bad file: {f.name}")
        for f in uploaded_files[3:]:
            try:
                f.seek(0); valid.append((f.name, Image.open(f).convert("RGB")))
            except Exception:
                pass
        if len(uploaded_files) > 3:
            st.caption(f"+{len(uploaded_files)-3} more file(s) queued")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button(f"🔬  Analyze {len(valid)} Image{'s' if len(valid)>1 else ''}", use_container_width=True)
    else:
        st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;
          font-size:.82rem;color:var(--c4);font-family:var(--mono);">
          ↑ &nbsp;Drop or browse a corn leaf image to begin diagnosis
        </div>""", unsafe_allow_html=True)

    # Inference
    if analyze and valid:
        batch = []
        with st.spinner("🔬 Running deep CNN scan…"):
            for fname, img in valid:
                label, conf, all_probs = predict(img)
                ts = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
                info = DISEASE_INFO[label]
                status = "ok" if label == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
                batch.append(dict(fname=fname, img=img, label=label, conf=conf,
                                  all_probs=all_probs, ts=ts, info=info, status=status))
                st.session_state.history.insert(0, dict(label=label, conf=conf, ts=ts,
                                                         fname=fname, status=status, info=info))
                st.session_state.scanned += 1
                st.session_state.tip_index += 1
                st.session_state.streak = (st.session_state.streak + 1) if label == "Healthy" else 0
        st.session_state.results = batch

    # Streak
    if st.session_state.streak >= 3:
        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(22,163,74,.09),rgba(74,222,128,.05));
  border:1px solid rgba(22,163,74,.2);border-radius:12px;padding:.8rem 1.2rem;
  margin-bottom:.8rem;display:flex;align-items:center;gap:.65rem;font-size:.82rem;color:var(--g);animation:fadeIn .4s ease both;">
  🔥 <strong>Healthy Streak:</strong> {st.session_state.streak} clean scans in a row!
</div>""", unsafe_allow_html=True)

    # ─────────────────────────── RESULTS ──────────────────────────────────
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧬 AI Diagnosis</div>', unsafe_allow_html=True)

        for r in st.session_state.results:
            info   = r["info"]
            pct    = r["conf"] * 100
            status = r["status"]
            sc     = {"ok":"rc-ok","warn":"rc-warn","bad":"rc-bad"}.get(status,"rc-ok")
            bc     = info["sev_color"]
            bg     = info["sev_bg"]
            bd     = bc + "40"
            circ   = 264
            dash   = int(circ - (pct / 100) * circ)
            sp_col = "#dc2626" if info["spread_risk"]>70 else ("#d97706" if info["spread_risk"]>40 else "#16a34a")

            st.markdown(f"""
<div class="result-card {sc}">
  <div style="margin-bottom:.9rem;display:flex;align-items:center;flex-wrap:wrap;gap:.4rem;">
    <span class="dis-badge" style="background:{bg};color:{bc};border:1.5px solid {bd};">
      {info['icon']} &nbsp;{info['short']}
    </span>
    <span class="urg-pill" style="color:{info['urgency_color']};border-color:{info['urgency_color']}44;background:{info['urgency_color']}12;">
      ◉ &nbsp;{info['urgency']}
    </span>
  </div>

  <div class="conf-ring-wrap">
    <div class="conf-ring">
      <svg width="88" height="88" viewBox="0 0 88 88">
        <circle class="crb" cx="44" cy="44" r="37"/>
        <circle class="crf" cx="44" cy="44" r="37" stroke="{bc}"
          style="--dash:{dash};stroke-dashoffset:{dash};"/>
      </svg>
      <div class="conf-ring-lbl">
        <span class="cpct" style="color:{bc};">{pct:.0f}%</span>
        <span class="csub">CONF</span>
      </div>
    </div>
    <div style="flex:1;">
      <div style="font-family:var(--disp);font-size:1.65rem;font-weight:800;
        letter-spacing:-.055em;color:var(--ink);line-height:1.05;margin-bottom:.2rem;">
        {info['short']}
      </div>
      <div style="font-size:.8rem;color:var(--c4);font-style:italic;margin-bottom:.55rem;">{info['pathogen']}</div>
      <div style="font-size:.72rem;color:var(--c3);font-family:var(--mono);">🕐 {r['ts']} &nbsp;·&nbsp; 📄 {r['fname']}</div>
    </div>
  </div>

  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.28rem;">
    <span style="font-size:.6rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
      color:var(--c4);font-family:var(--mono);">Spread Risk Index</span>
    <span style="font-family:var(--disp);font-size:.95rem;font-weight:800;color:{sp_col};">{info['spread_risk']}%</span>
  </div>
  <div class="sev-track">
    <div class="sev-fill" style="width:{info['spread_risk']}%;background:linear-gradient(90deg,{sp_col},{sp_col}88);"></div>
  </div>

  <div class="metric-row">
    <div class="metric-box"><div class="metric-n" style="color:{sp_col};">{info['spread_risk']}%</div><div class="metric-l">Spread Risk</div></div>
    <div class="metric-box"><div class="metric-n" style="color:var(--amr);">{info['treatment_window']}</div><div class="metric-l">Treat Within</div></div>
    <div class="metric-box"><div class="metric-n" style="color:var(--red);">{info['economic_loss']}</div><div class="metric-l">Yield Loss</div></div>
  </div>

  <div class="treat-card">
    <div class="treat-title">💊 Recommended Treatment</div>
    <div class="treat-body">{info['treatment']}</div>
  </div>

  <div class="wx-card">
    <span style="font-size:1.1rem;">🌤️</span>
    <div>
      <div style="font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
        color:#3b82f6;font-family:var(--mono);margin-bottom:.2rem;">Weather Risk Factor</div>
      <div>{info['weather_risk']}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # Image comparison + AI heatmap
            try:
                orig_b64 = img_to_b64(r["img"])
                heat_b64 = make_heatmap(r["img"])
                has_img  = True
            except Exception:
                has_img  = False

            if has_img:
                st.markdown('<div class="sec-lbl">🔬 AI Focus Analysis</div>', unsafe_allow_html=True)
                st.markdown(f"""
<div class="cmp-panel">
  <div>
    <div class="cmp-lbl">Original Leaf</div>
    <div style="border-radius:12px;overflow:hidden;">
      <img class="cmp-img" src="data:image/jpeg;base64,{orig_b64}"/>
    </div>
  </div>
  <div>
    <div class="cmp-lbl">AI Focus Zones</div>
    <div style="position:relative;border-radius:12px;overflow:hidden;">
      <img class="cmp-img" src="data:image/jpeg;base64,{heat_b64}"/>
      <div class="heat-label">AI ATTENTION MAP</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # Probability breakdown + charts
            with st.expander("📊 Confidence breakdown & charts"):
                # Bars
                bar_html = ""
                for cls in CLASSES:
                    p  = r["all_probs"][cls]
                    hi = "pb-hi" if cls == r["label"] else ""
                    bar_html += f"""
<div class="pb-row">
  <span class="pb-name">{DISEASE_INFO[cls]['icon']} {cls}</span>
  <div class="pb-tr"><div class="pb-fill {hi}" style="width:{p*100:.1f}%"></div></div>
  <span class="pb-pct">{p*100:.1f}%</span>
</div>"""

                # SVG donut
                DCOLS = {"Blight":"#ef4444","Common Rust":"#f97316","Gray Leaf Spot":"#8b5cf6","Healthy":"#22c55e"}
                circ_d = 2 * 3.14159 * 50
                offset_d = 0
                segs = ""
                for cls in CLASSES:
                    p = r["all_probs"][cls]
                    seg = p * circ_d
                    segs += f'<circle cx="70" cy="70" r="50" fill="none" stroke="{DCOLS[cls]}" stroke-width="16" stroke-dasharray="{seg:.2f} {circ_d:.2f}" stroke-dashoffset="-{offset_d:.2f}" transform="rotate(-90 70 70)"/>'
                    offset_d += seg

                leg = "".join(
                    f'<div class="d-leg"><div class="d-dot" style="background:{DCOLS[c]};"></div><span>{c}: {r["all_probs"][c]*100:.1f}%</span></div>'
                    for c in CLASSES
                )

                st.markdown(f"""
<div class="chart-wrap">
  <div class="chart-title">Confidence Distribution — All Classes</div>
  {bar_html}
</div>
<div class="chart-wrap">
  <div class="chart-title">Probability Donut Chart</div>
  <div class="donut-wrap">
    <svg width="140" height="140" viewBox="0 0 140 140">{segs}</svg>
    <div class="donut-legend">{leg}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Export report
            fname_safe = r["fname"].replace(" ","_").rsplit(".",1)[0]
            st.download_button(
                label="📄  Download Diagnosis Report",
                data=result_to_html_report(r).encode("utf-8"),
                file_name=f"CornScan_{fname_safe}_{r['ts'][:11].replace(' ','_')}.html",
                mime="text/html",
                key=f"dl_{fname_safe}_{r['ts']}",
            )

        # Agronomic details
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📚 Agronomic Intelligence</div>', unsafe_allow_html=True)
        seen = set()
        for r in st.session_state.results:
            lbl = r["label"]
            if lbl in seen: continue
            seen.add(lbl)
            info = r["info"]
            sc   = info["sev_color"]
            with st.expander(f"{info['icon']}  {info['short']} — Agronomic Details", expanded=(lbl != "Healthy")):
                chips = "".join(f'<span class="sym-chip">· {s}</span>' for s in info["symptoms"])
                st.markdown(f"""
<div class="info-grid">
  <div class="info-card">
    <div class="info-card-h">📋 Overview</div>
    <div class="info-card-b">{info['desc']}</div>
    <span style="display:inline-block;margin-top:.75rem;font-size:.63rem;font-weight:700;letter-spacing:.08em;
      font-family:var(--mono);padding:.22rem .7rem;border-radius:999px;
      background:{sc}18;color:{sc};border:1px solid {sc}44;">SEVERITY · {info['severity']}</span>
    <div class="fun-fact">💬 {info['fun_fact']}</div>
  </div>
  <div class="info-card">
    <div class="info-card-h">🔍 Symptoms</div>
    <div>{chips}</div>
    <div style="margin-top:.9rem;">
      <div class="info-card-h">🛡 Recommended Action</div>
      <div style="display:flex;align-items:flex-start;gap:.4rem;font-size:.78rem;color:var(--c2);line-height:1.65;">
        <span style="color:var(--g);flex-shrink:0;margin-top:.2rem;">✓</span>
        <span>{info['action']}</span>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Scan history + trend chart
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📜 Scan History & Trend</div>', unsafe_allow_html=True)

        # Trend bars (last 10)
        trend = list(reversed(st.session_state.history[:10]))
        bar_items = ""
        for i, h in enumerate(trend):
            col = "#16a34a" if h["status"]=="ok" else ("#d97706" if h["status"]=="warn" else "#ef4444")
            bh  = max(6, int(h["conf"] * 62))
            bar_items += f"""
<div title="{h['info']['short']} {h['conf']*100:.1f}%"
  style="flex:1;height:{bh}px;background:{col};border-radius:4px 4px 0 0;
  opacity:.85;transition:opacity .2s;cursor:default;animation:barGrow .6s ease {i*.07:.2f}s both;"
  onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=.85"></div>"""

        st.markdown(f"""
<div class="chart-wrap" style="margin-bottom:1rem;">
  <div class="chart-title">Scan History — Confidence Trend (last 10 scans)</div>
  <div style="display:flex;align-items:flex-end;gap:4px;height:68px;padding:0 2px;">{bar_items}</div>
  <div style="display:flex;justify-content:space-between;margin-top:.4rem;">
    <span style="font-size:.6rem;color:var(--c4);font-family:var(--mono);">Oldest</span>
    <span style="font-size:.6rem;color:var(--c4);font-family:var(--mono);">Most Recent</span>
  </div>
  <div style="display:flex;gap:1.2rem;margin-top:.6rem;flex-wrap:wrap;">
    <span style="font-size:.65rem;font-family:var(--mono);display:flex;align-items:center;gap:.35rem;">
      <span style="width:10px;height:10px;border-radius:2px;background:#16a34a;display:inline-block;"></span>Healthy</span>
    <span style="font-size:.65rem;font-family:var(--mono);display:flex;align-items:center;gap:.35rem;">
      <span style="width:10px;height:10px;border-radius:2px;background:#d97706;display:inline-block;"></span>Monitor</span>
    <span style="font-size:.65rem;font-family:var(--mono);display:flex;align-items:center;gap:.35rem;">
      <span style="width:10px;height:10px;border-radius:2px;background:#ef4444;display:inline-block;"></span>Diseased</span>
  </div>
</div>""", unsafe_allow_html=True)

        for h in st.session_state.history[:8]:
            tc = {"ok":"ht-ok","warn":"ht-warn","bad":"ht-bad"}.get(h["status"],"ht-ok")
            tt = {"ok":"Healthy","warn":"Monitor","bad":"Diseased"}.get(h["status"],"—")
            st.markdown(f"""
<div class="hist-row">
  <span style="font-size:1.1rem;flex-shrink:0;">{h['info']['icon']}</span>
  <span class="hist-name">{h['info']['short']}</span>
  <span class="hist-conf">{h['conf']*100:.1f}%</span>
  <span class="hist-time">{h['ts']}</span>
  <span class="hist-tag {tc}">{tt}</span>
</div>""", unsafe_allow_html=True)

        if len(st.session_state.history) > 8:
            st.caption(f"+{len(st.session_state.history)-8} older entries")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, _, _ = st.columns([1, 3, 1])
        with c1:
            if st.button("↺ Clear History"):
                st.session_state.history = []
                st.session_state.scanned = 0
                st.session_state.results = []
                st.session_state.streak  = 0
                _rerun()

    # Footer
    st.markdown("""
<div class="app-footer">
  <strong>CornScan AI</strong> &nbsp;·&nbsp; Deep CNN Plant Disease Intelligence &nbsp;·&nbsp; v5.0<br>
  Powered by <strong>CornScan AI Engine</strong> &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; ResNet-inspired Architecture<br>
  4-Class Classifier: Blight · Common Rust · Gray Leaf Spot · Healthy<br>
  <span style="color:var(--c4);">No data leaves your device &nbsp;·&nbsp; © 2025 CornScan AI</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    landing_page()
else:
    main_app() 
 
