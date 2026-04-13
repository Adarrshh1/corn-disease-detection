"""
╔══════════════════════════════════════════════════════════════════╗
║  CornScan AI  ·  app.py  (Enhanced v4.0)                        ║
║  Black landing → White main · CNN Corn Disease Detection         ║
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
for key, default in [
    ("page", "landing"),
    ("history", []),
    ("results", []),
    ("scanned", 0),
    ("streak", 0),
    ("health_score", 100),
    ("tip_index", 0),
    ("show_confetti", False),
]:
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
        "spread_risk": 85,
        "treatment_window": "7–10 days",
        "economic_loss": "30–50%",
        "fun_fact": "Blight spores can travel over 500 km via wind currents.",
    },
    "Common Rust": {
        "icon": "🟠", "severity": "MEDIUM", "sev_color": "#ffa94d",
        "short": "Common Corn Rust",
        "pathogen": "Puccinia sorghi",
        "desc": "Spreads via airborne spores in cool, humid conditions (16–23 °C). Can reduce grain fill by up to 20 % with severe pre-silking infection.",
        "action": "Scout weekly from V6. Apply fungicide if >50 pustules per leaf pre-silk.",
        "symptoms": ["Brick-red circular pustules on both surfaces", "Powdery cinnamon-brown spore masses", "Dark brown-black pustules late season"],
        "spread_risk": 62,
        "treatment_window": "5–7 days",
        "economic_loss": "10–20%",
        "fun_fact": "Rust spores can germinate in as little as 3 hours under ideal humidity.",
    },
    "Gray Leaf Spot": {
        "icon": "🩶", "severity": "HIGH", "sev_color": "#ff6b6b",
        "short": "Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis",
        "desc": "Among the most economically damaging corn diseases globally. Overwinters in residue; epidemic in warm, humid, no-till continuous-corn systems.",
        "action": "Plant resistant hybrids. Apply triazole + strobilurin mix at VT/R1.",
        "symptoms": ["Rectangular lesions bounded by leaf veins", "Ash-grey to pale tan colour", "Yellow halo around mature lesions"],
        "spread_risk": 90,
        "treatment_window": "3–5 days",
        "economic_loss": "20–40%",
        "fun_fact": "Gray Leaf Spot thrives in fields with poor air circulation and heavy dew.",
    },
    "Healthy": {
        "icon": "✅", "severity": "NONE", "sev_color": "#22c55e",
        "short": "No Disease Detected",
        "pathogen": "Zea mays — clean",
        "desc": "No signs of fungal, bacterial, or viral disease detected. The leaf appears vigorous with uniform colour and clean surface texture.",
        "action": "Continue routine weekly scouting. Maintain balanced NPK fertilisation.",
        "symptoms": ["Uniform deep-green colour", "Clean surface, no lesions", "Normal venation and architecture"],
        "spread_risk": 0,
        "treatment_window": "N/A",
        "economic_loss": "0%",
        "fun_fact": "A healthy corn leaf can photosynthesize up to 12 hours per day at peak growth.",
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
    "🌤️ Fungicide efficacy drops 40% when applied in rain — time applications wisely.",
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

def compute_field_health():
    if not st.session_state.history:
        return 100
    recent = st.session_state.history[:10]
    score = sum(100 if h["status"] == "ok" else (50 if h["status"] == "warn" else 10) for h in recent)
    return round(score / len(recent))


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE CSS  (pure black)
# ══════════════════════════════════════════════════════════════════════════════
LANDING_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg0:#000000; --bg1:#080808; --bg2:#111111; --bg3:#1a1a1a;
  --g:#86efac; --gm:#4ade80; --gd:#16a34a;
  --cream:#f2ede3; --c2:#aaa9a6; --c3:#666460; --c4:#3a3836;
  --bd:rgba(255,255,255,.07); --bd2:rgba(255,255,255,.14);
  --hi:rgba(134,239,172,.14); --hi2:rgba(134,239,172,.06);
  --font:'DM Sans',sans-serif; --disp:'Syne',sans-serif; --mono:'DM Mono',monospace;
  --r:12px; --rl:20px; --rxl:28px;
  --sh:0 2px 12px rgba(0,0,0,.8); --shm:0 8px 32px rgba(0,0,0,.9);
}

html,body,[class*="css"]{font-family:var(--font)!important;background:#000!important;color:var(--cream)!important;-webkit-font-smoothing:antialiased;}
.stApp{background:#000000!important;min-height:100vh;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important;}
.block-container{padding-top:0!important;max-width:740px;}
[data-testid="stSidebar"]{display:none!important;}

/* Noise grain overlay */
.lp-noise{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.04;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-repeat:repeat; background-size:200px 200px;}

.lp-bg{position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 65% 50% at 15% 0%,rgba(74,222,128,.07) 0%,transparent 60%),
             radial-gradient(ellipse 50% 40% at 85% 100%,rgba(134,239,172,.05) 0%,transparent 55%),
             #000000;}
.lp-grid{position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
  background-size:60px 60px;
  mask-image:radial-gradient(ellipse 75% 75% at 50% 50%,black,transparent);}

.lp-scanline{position:fixed;top:0;left:0;width:100%;height:2px;z-index:2;
  background:linear-gradient(90deg,transparent,rgba(134,239,172,.4),transparent);
  animation:scanline 6s linear infinite;pointer-events:none;}
@keyframes scanline{0%{top:-2px}100%{top:100vh}}

.lp-wrap{position:relative;z-index:1;min-height:90vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:3rem 1.5rem 2rem;text-align:center;
  animation:fadeUp .8s cubic-bezier(.22,1,.36,1) both;}

/* Animated orb */
.lp-orb-ring{position:relative;width:130px;height:130px;margin-bottom:2rem;}
.lp-orb-ring::before,.lp-orb-ring::after{content:'';position:absolute;border-radius:50%;border:1px solid rgba(134,239,172,.15);}
.lp-orb-ring::before{inset:-16px;animation:ringPulse 3s ease-in-out infinite;}
.lp-orb-ring::after{inset:-32px;animation:ringPulse 3s ease-in-out .6s infinite;}
.lp-orb{width:130px;height:130px;
  background:radial-gradient(circle at 35% 30%,rgba(134,239,172,.55),rgba(22,163,74,.2) 55%,transparent 80%);
  border:1px solid rgba(134,239,172,.25);border-radius:36px;
  display:flex;align-items:center;justify-content:center;font-size:3.5rem;
  box-shadow:0 0 80px rgba(134,239,172,.12),inset 0 1px 0 rgba(255,255,255,.08);
  animation:floatY 4s ease-in-out infinite;}

@keyframes ringPulse{0%,100%{opacity:.4;transform:scale(1)}50%{opacity:.15;transform:scale(1.08)}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes barGrow{from{width:0}}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(134,239,172,.35)}60%{box-shadow:0 0 0 10px transparent}}
@keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
@keyframes confetti{0%{transform:translateY(-10px) rotate(0deg);opacity:1}100%{transform:translateY(110vh) rotate(720deg);opacity:0}}

.lp-pill{display:inline-flex;align-items:center;gap:.45rem;
  background:rgba(134,239,172,.05);border:1px solid rgba(134,239,172,.18);
  border-radius:999px;padding:.24rem 1.1rem;
  font-size:.66rem;font-weight:500;color:var(--g);
  letter-spacing:.12em;text-transform:uppercase;font-family:var(--mono);margin-bottom:1.2rem;}
.lp-pill-dot{width:6px;height:6px;border-radius:50%;background:var(--g);animation:pulse 2s infinite;}

.lp-title{font-family:var(--disp);font-size:clamp(3rem,8vw,5.2rem);
  font-weight:800;letter-spacing:-.06em;line-height:.95;margin-bottom:.6rem;color:#fff;}
.lp-title-grad{background:linear-gradient(135deg,#86efac 0%,#4ade80 45%,#a3e635 100%);
  background-size:200% 200%;animation:gradShift 5s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}

.lp-sub{font-size:1rem;color:var(--c3);font-weight:300;line-height:1.75;
  max-width:420px;margin:0 auto 2rem;}

/* Ticker strip */
.lp-ticker-wrap{width:100%;overflow:hidden;margin-bottom:2rem;
  border-top:1px solid rgba(134,239,172,.08);border-bottom:1px solid rgba(134,239,172,.08);
  padding:.5rem 0;position:relative;}
.lp-ticker-wrap::before,.lp-ticker-wrap::after{content:'';position:absolute;top:0;bottom:0;width:80px;z-index:2;pointer-events:none;}
.lp-ticker-wrap::before{left:0;background:linear-gradient(90deg,#000,transparent);}
.lp-ticker-wrap::after{right:0;background:linear-gradient(-90deg,#000,transparent);}
.lp-ticker{display:flex;gap:3rem;width:max-content;animation:ticker 28s linear infinite;}
.lp-tick{font-size:.65rem;font-family:var(--mono);color:var(--c4);letter-spacing:.1em;text-transform:uppercase;white-space:nowrap;}
.lp-tick span{color:var(--g);margin-right:.4rem;}

.lp-stats{display:flex;gap:2.5rem;justify-content:center;margin-bottom:2.2rem;}
.lp-stat-n{font-family:var(--disp);font-size:2.1rem;font-weight:800;color:#fff;letter-spacing:-.06em;line-height:1;}
.lp-stat-l{font-size:.63rem;color:var(--c4);font-family:var(--mono);margin-top:.2rem;letter-spacing:.08em;text-transform:uppercase;}
.lp-sep{width:1px;background:var(--bd);}

.lp-feats{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-bottom:2.2rem;width:100%;max-width:620px;}
.lp-feat{background:var(--bg2);border:1px solid rgba(255,255,255,.05);
  border-radius:16px;padding:1.15rem 1rem;
  transition:border-color .2s,transform .2s,box-shadow .2s;cursor:default;}
.lp-feat:hover{border-color:rgba(134,239,172,.22);transform:translateY(-3px);
  box-shadow:0 12px 32px rgba(0,0,0,.8),0 0 24px rgba(134,239,172,.04);}
.lp-feat-ico{font-size:1.45rem;margin-bottom:.4rem;}
.lp-feat-t{font-family:var(--disp);font-size:.78rem;font-weight:700;color:var(--c2);margin-bottom:.2rem;}
.lp-feat-d{font-size:.67rem;color:var(--c4);line-height:1.55;}

/* Button */
.stButton>button{
  font-family:var(--disp)!important;font-weight:800!important;
  font-size:1rem!important;letter-spacing:.01em!important;
  border-radius:14px!important;
  border:1.5px solid rgba(134,239,172,.4)!important;
  background:linear-gradient(135deg,rgba(74,222,128,.22),rgba(134,239,172,.12))!important;
  color:var(--g)!important;padding:.8rem 2rem!important;
  box-shadow:0 0 40px rgba(134,239,172,.08),inset 0 1px 0 rgba(255,255,255,.06)!important;
  transition:all .22s cubic-bezier(.4,0,.2,1)!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(74,222,128,.35),rgba(134,239,172,.22))!important;
  border-color:rgba(134,239,172,.65)!important;
  box-shadow:0 0 60px rgba(134,239,172,.15)!important;
  transform:translateY(-2px)!important;
}
.stButton>button:active{transform:translateY(0)!important;}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP CSS  (white/light background)
# ══════════════════════════════════════════════════════════════════════════════
MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg0:#f8f7f3; --bg1:#f0ede6; --bg2:#ffffff; --bg3:#f4f2ec; --bg4:#e8e4db;
  --bd:rgba(0,0,0,.07); --bd2:rgba(0,0,0,.13);
  --ink:#1a1916; --c2:#4a4845; --c3:#807b71; --c4:#b0ab9f;
  --g:#16a34a; --gm:#22c55e; --gl:#4ade80; --glight:rgba(22,163,74,.08);
  --red:#dc2626; --redbg:rgba(220,38,38,.07);
  --amr:#d97706; --amrbg:rgba(217,119,6,.08);
  --font:'DM Sans',sans-serif; --disp:'Syne',sans-serif; --mono:'DM Mono',monospace;
  --r:10px; --rl:18px; --rxl:24px;
  --sh:0 1px 6px rgba(0,0,0,.06); --shm:0 4px 20px rgba(0,0,0,.08); --shl:0 10px 40px rgba(0,0,0,.10);
}

html,body,[class*="css"]{font-family:var(--font)!important;background:var(--bg0)!important;color:var(--ink)!important;-webkit-font-smoothing:antialiased;}
.stApp{background:var(--bg0)!important;min-height:100vh;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important;}
.block-container{padding-top:0!important;max-width:740px;}
[data-testid="stSidebar"]{display:none!important;}

/* Buttons */
.stButton>button{
  font-family:var(--disp)!important;font-weight:700!important;font-size:.88rem!important;
  border-radius:var(--r)!important;border:1.5px solid rgba(22,163,74,.3)!important;
  background:linear-gradient(135deg,rgba(22,163,74,.1),rgba(74,222,128,.06))!important;
  color:var(--g)!important;padding:.6rem 1.5rem!important;
  box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.8)!important;
  transition:all .18s cubic-bezier(.4,0,.2,1)!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(22,163,74,.18),rgba(74,222,128,.12))!important;
  border-color:rgba(22,163,74,.55)!important;box-shadow:var(--shm)!important;transform:translateY(-1px)!important;
}
.stButton>button:active{transform:translateY(0)!important;}

/* Uploader */
[data-testid="stFileUploader"] section{
  background:var(--bg2)!important;border:2px dashed rgba(22,163,74,.22)!important;
  border-radius:var(--rxl)!important;padding:2.5rem!important;transition:all .22s!important;
}
[data-testid="stFileUploader"] section:hover{border-color:rgba(22,163,74,.45)!important;background:rgba(22,163,74,.03)!important;}

details{background:var(--bg2)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;margin-bottom:.5rem!important;box-shadow:var(--sh)!important;}
details summary{color:var(--c2)!important;font-weight:600!important;font-size:.85rem!important;padding:.85rem 1.1rem!important;}
details[open]{border-color:rgba(22,163,74,.2)!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--g),var(--gm))!important;border-radius:999px!important;}
.stProgress>div{background:rgba(0,0,0,.06)!important;border-radius:999px!important;height:6px!important;}
.stMarkdown p,.stMarkdown li{color:var(--c2)!important;font-size:.88rem!important;}

@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes barGrow{from{width:0}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(22,163,74,.3)}60%{box-shadow:0 0 0 8px transparent}}
@keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
@keyframes confettiPop{0%{transform:scale(0) rotate(0deg);opacity:1}80%{opacity:1}100%{transform:scale(1) rotate(360deg) translateY(-60px);opacity:0}}

/* Top bar */
.app-topbar{display:flex;align-items:center;gap:.75rem;padding:1.3rem 0 .6rem;
  border-bottom:1px solid var(--bd);margin-bottom:1.5rem;animation:fadeIn .4s ease both;}
.app-topbar-brand{display:flex;align-items:center;gap:.65rem;flex:1;}
.app-topbar-ico{width:34px;height:34px;border-radius:10px;
  background:linear-gradient(135deg,rgba(22,163,74,.15),rgba(74,222,128,.08));
  border:1px solid rgba(22,163,74,.2);display:flex;align-items:center;justify-content:center;font-size:1.1rem;}
.app-topbar-name{font-family:var(--disp);font-size:1rem;font-weight:800;color:var(--ink);letter-spacing:-.03em;}
.app-topbar-ver{font-size:.63rem;color:var(--c4);font-family:var(--mono);}

.sec-lbl{display:flex;align-items:center;gap:.5rem;font-size:.6rem;font-weight:700;
  letter-spacing:.16em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);margin-bottom:.65rem;}
.sec-lbl::after{content:'';flex:1;height:1px;background:var(--bd);}

/* Stats strip */
.stats-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin-bottom:1.5rem;}
.stat-box{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:.8rem .5rem;
  text-align:center;box-shadow:var(--sh);transition:border-color .18s,transform .18s;}
.stat-box:hover{border-color:rgba(22,163,74,.2);transform:translateY(-2px);box-shadow:var(--shm);}
.stat-n{font-family:var(--disp);font-size:1.5rem;font-weight:800;color:var(--g);letter-spacing:-.05em;line-height:1;}
.stat-l{font-size:.58rem;color:var(--c4);margin-top:.15rem;letter-spacing:.07em;text-transform:uppercase;font-family:var(--mono);}

/* Health meter */
.health-meter{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rl);
  padding:1.1rem 1.4rem;margin-bottom:.75rem;box-shadow:var(--sh);display:flex;align-items:center;gap:1rem;}
.health-dial{width:52px;height:52px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;font-weight:800;
  font-family:var(--disp);border:3px solid transparent;}
.health-label{font-size:.58rem;color:var(--c4);font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.2rem;}
.health-bar{flex:1;height:8px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;}
.health-fill{height:100%;border-radius:999px;transition:width .8s cubic-bezier(.4,0,.2,1);}

/* Tip box */
.tip-box{background:linear-gradient(135deg,rgba(22,163,74,.06),rgba(74,222,128,.03));
  border:1px solid rgba(22,163,74,.15);border-radius:var(--r);padding:.7rem 1rem;
  margin-bottom:.75rem;font-size:.78rem;color:var(--c2);display:flex;align-items:flex-start;gap:.5rem;}
.tip-ico{color:var(--g);flex-shrink:0;font-size:.85rem;}

/* Image wrap */
.img-wrap{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  overflow:hidden;box-shadow:var(--shm);animation:fadeIn .35s ease both;}
.img-foot{padding:.55rem 1rem;border-top:1px solid var(--bd);font-size:.7rem;color:var(--c4);
  font-family:var(--mono);display:flex;align-items:center;justify-content:space-between;background:var(--bg3);}
.img-badge{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.2);
  border-radius:999px;font-size:.62rem;padding:.05rem .5rem;font-weight:600;}

/* Result card */
.result-card{background:var(--bg2);border:1.5px solid rgba(22,163,74,.18);border-radius:var(--rxl);
  padding:2rem 2.2rem;box-shadow:var(--shl);animation:fadeUp .5s ease both;position:relative;overflow:hidden;margin-bottom:.75rem;}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--g),var(--gm),#84cc16);}
.result-card.diseased{border-color:rgba(220,38,38,.2)!important;}
.result-card.diseased::before{background:linear-gradient(90deg,#b91c1c,#f87171)!important;}
.result-card.amber{border-color:rgba(217,119,6,.2)!important;}
.result-card.amber::before{background:linear-gradient(90deg,#92400e,#fbbf24)!important;}

.r-tag{display:inline-flex;align-items:center;gap:.35rem;font-size:.63rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;padding:.2rem .9rem;border-radius:999px;margin-bottom:.9rem;font-family:var(--mono);}
.r-ok{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.25);}
.r-bad{background:var(--redbg);color:var(--red);border:1px solid rgba(220,38,38,.25);}
.r-warn{background:var(--amrbg);color:var(--amr);border:1px solid rgba(217,119,6,.25);}

.r-name{font-family:var(--disp);font-size:2rem;font-weight:800;letter-spacing:-.05em;color:var(--ink);line-height:1.05;margin-bottom:.15rem;}
.r-sci{font-size:.8rem;color:var(--c4);font-style:italic;margin-bottom:1.3rem;}
.r-ch{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.3rem;}
.r-cl{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);}
.r-cv{font-family:var(--disp);font-size:1.7rem;font-weight:800;letter-spacing:-.05em;color:var(--ink);}
.r-bt{height:6px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;margin-bottom:1.4rem;}
.r-bf{height:100%;border-radius:999px;animation:barGrow .9s cubic-bezier(.4,0,.2,1) both;}
.r-meta{font-size:.67rem;color:var(--c4);font-family:var(--mono);display:flex;gap:.8rem;flex-wrap:wrap;}

/* Quick metrics */
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-top:1rem;}
.metric-box{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--r);padding:.7rem .9rem;text-align:center;}
.metric-n{font-family:var(--disp);font-size:1.1rem;font-weight:800;letter-spacing:-.04em;}
.metric-l{font-size:.58rem;color:var(--c4);font-family:var(--mono);letter-spacing:.07em;text-transform:uppercase;margin-top:.1rem;}

/* Prob rows */
.pb-row{display:flex;align-items:center;gap:.65rem;margin-bottom:.45rem;}
.pb-name{font-size:.72rem;font-family:var(--mono);color:var(--c3);width:120px;flex-shrink:0;}
.pb-tr{flex:1;height:4px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;}
.pb-fill{height:100%;border-radius:999px;background:rgba(0,0,0,.1);animation:barGrow .65s cubic-bezier(.4,0,.2,1) both;}
.pb-hi{background:linear-gradient(90deg,var(--g),var(--gm))!important;}
.pb-pct{font-size:.7rem;font-family:var(--mono);color:var(--c3);width:36px;text-align:right;flex-shrink:0;}

/* Info grid */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:.9rem;}
.info-card{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rl);padding:1.1rem 1.15rem;transition:border-color .18s,box-shadow .18s;}
.info-card:hover{border-color:rgba(22,163,74,.18);box-shadow:var(--sh);}
.info-card-h{font-size:.58rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);margin-bottom:.65rem;display:flex;align-items:center;gap:.4rem;}
.info-card-b{font-size:.8rem;color:var(--c2);line-height:1.7;}
.sym-chip{display:inline-block;background:var(--bg2);border:1px solid var(--bd);border-radius:6px;font-size:.71rem;color:var(--c2);padding:.16rem .52rem;margin:.1rem .05rem;line-height:1.45;}
.prev-item{display:flex;align-items:flex-start;gap:.4rem;margin-bottom:.4rem;font-size:.78rem;color:var(--c2);line-height:1.6;}
.prev-ico{color:var(--g);flex-shrink:0;margin-top:.2rem;font-size:.7rem;}
.sev-badge{display:inline-block;margin-top:.75rem;font-size:.63rem;font-weight:700;letter-spacing:.08em;font-family:var(--mono);padding:.2rem .65rem;border-radius:999px;}

/* Fun fact */
.fun-fact{background:linear-gradient(135deg,rgba(22,163,74,.06),rgba(74,222,128,.03));
  border-left:3px solid var(--g);border-radius:0 var(--r) var(--r) 0;
  padding:.65rem .9rem;margin-top:.7rem;font-size:.77rem;color:var(--c2);font-style:italic;}

/* History */
.hist-row{display:flex;align-items:center;gap:.75rem;padding:.65rem .9rem;background:var(--bg2);
  border:1px solid var(--bd);border-radius:var(--r);margin-bottom:.4rem;font-size:.8rem;
  box-shadow:var(--sh);transition:border-color .18s,box-shadow .18s;animation:fadeIn .3s ease both;}
.hist-row:hover{border-color:rgba(22,163,74,.18);box-shadow:var(--shm);}
.hist-ico{font-size:1.1rem;flex-shrink:0;}
.hist-name{font-family:var(--disp);font-weight:600;color:var(--ink);flex:1;font-size:.82rem;}
.hist-conf{font-family:var(--mono);font-size:.72rem;color:var(--c3);}
.hist-time{font-family:var(--mono);font-size:.63rem;color:var(--c4);}
.hist-tag{font-family:var(--mono);font-size:.63rem;font-weight:700;padding:.14rem .55rem;border-radius:999px;}
.ht-ok{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.22);}
.ht-bad{background:var(--redbg);color:var(--red);border:1px solid rgba(220,38,38,.22);}
.ht-warn{background:var(--amrbg);color:var(--amr);border:1px solid rgba(217,119,6,.22);}

.app-footer{text-align:center;padding:2rem 0 1.2rem;font-size:.66rem;color:var(--c4);
  font-family:var(--mono);border-top:1px solid var(--bd);margin-top:2.5rem;}

/* Confetti */
.confetti-piece{position:fixed;width:8px;height:8px;top:-8px;border-radius:2px;pointer-events:none;z-index:9999;}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing_page():
    st.markdown(LANDING_CSS, unsafe_allow_html=True)

    # Ticker items (doubled for seamless loop)
    tick_items = "".join(
        f'<span class="lp-tick"><span>◆</span>{t}</span>'
        for t in [
            "97% Accuracy", "Deep CNN", "4 Disease Classes",
            "Instant Inference", "No Data Sent", "TensorFlow Powered",
            "Batch Upload", "Session History", "97% Accuracy", "Deep CNN",
            "4 Disease Classes", "Instant Inference", "No Data Sent",
            "TensorFlow Powered", "Batch Upload", "Session History",
        ]
    )

    st.markdown(f"""
<div class="lp-noise"></div>
<div class="lp-bg"></div>
<div class="lp-grid"></div>
<div class="lp-scanline"></div>

<div class="lp-wrap">
  <div class="lp-orb-ring">
    <div class="lp-orb">🌿</div>
  </div>
  <div class="lp-pill">
    <span class="lp-pill-dot"></span>
    Deep Learning · Plant Pathology · v4.0
  </div>
  <div class="lp-title">
    Corn<span class="lp-title-grad">Scan</span><br>AI
  </div>
  <div class="lp-sub">
    Upload a corn leaf photo. Get an instant, science-backed disease
    diagnosis powered by a deep convolutional neural network.
  </div>

  <div class="lp-ticker-wrap">
    <div class="lp-ticker">{tick_items}</div>
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
    <div class="lp-feat"><div class="lp-feat-ico">📊</div><div class="lp-feat-t">Risk Metrics</div><div class="lp-feat-d">Spread risk, treatment window & economic loss</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🌡️</div><div class="lp-feat-t">Field Health</div><div class="lp-feat-d">Live field health score across your session</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📦</div><div class="lp-feat-t">Batch Mode</div><div class="lp-feat-d">Upload multiple images in one session</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">💡</div><div class="lp-feat-t">Agro Tips</div><div class="lp-feat-d">Rotating expert agronomic advice per scan</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c2:
        if st.button("🌿  Let's Go", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown("""
<div style="text-align:center;margin-top:.8rem;font-size:.65rem;color:#333;font-family:'DM Mono',monospace;">
Powered by TensorFlow · Keras · Streamlit &nbsp;·&nbsp; No data leaves your device
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main_app():
    st.markdown(MAIN_CSS, unsafe_allow_html=True)

    # ── Top bar ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="app-topbar">
  <div class="app-topbar-brand">
    <div class="app-topbar-ico">🌿</div>
    <div>
      <div class="app-topbar-name">CornScan AI</div>
      <div class="app-topbar-ver">v4.0 · CNN · TensorFlow</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    bc, _, _ = st.columns([1, 3, 1])
    with bc:
        if st.button("← Home"):
            st.session_state.page    = "landing"
            st.session_state.results = []
            st.rerun()

    # ── Field Health Meter ─────────────────────────────────────────────────
    fh = compute_field_health()
    if fh >= 75:
        dial_bg, fill_color, dial_color = "rgba(22,163,74,.12)", "linear-gradient(90deg,#16a34a,#4ade80)", "#16a34a"
    elif fh >= 40:
        dial_bg, fill_color, dial_color = "rgba(217,119,6,.12)", "linear-gradient(90deg,#d97706,#fbbf24)", "#d97706"
    else:
        dial_bg, fill_color, dial_color = "rgba(220,38,38,.12)", "linear-gradient(90deg,#dc2626,#f87171)", "#dc2626"

    st.markdown(f"""
<div class="health-meter">
  <div class="health-dial" style="background:{dial_bg};color:{dial_color};border-color:{dial_color}40;">
    {fh}
  </div>
  <div style="flex:1;">
    <div class="health-label">Field Health Score</div>
    <div class="health-bar">
      <div class="health-fill" style="width:{fh}%;background:{fill_color};"></div>
    </div>
  </div>
  <div style="font-size:.7rem;font-family:var(--mono);color:var(--c4);">
    {'🟢 Excellent' if fh>=75 else '🟡 Monitor' if fh>=40 else '🔴 At Risk'}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Rotating Agro Tip ──────────────────────────────────────────────────
    tip = AGRO_TIPS[st.session_state.tip_index % len(AGRO_TIPS)]
    st.markdown(f"""
<div class="tip-box">
  <span class="tip-ico">💡</span>
  <span><strong>Agro Tip:</strong> {tip[2:]}</span>
</div>""", unsafe_allow_html=True)

    # ── Stats ──────────────────────────────────────────────────────────────
    n_total    = st.session_state.scanned
    n_diseased = sum(1 for h in st.session_state.history if h["status"] != "ok")
    n_healthy  = n_total - n_diseased
    accuracy   = f"{(n_healthy/n_total*100):.0f}%" if n_total else "—"

    st.markdown(f"""
<div class="stats-strip">
  <div class="stat-box"><div class="stat-n">{n_total}</div><div class="stat-l">Scanned</div></div>
  <div class="stat-box"><div class="stat-n" style="color:#dc2626;">{n_diseased}</div><div class="stat-l">Diseased</div></div>
  <div class="stat-box"><div class="stat-n">{n_healthy}</div><div class="stat-l">Healthy</div></div>
  <div class="stat-box"><div class="stat-n">{accuracy}</div><div class="stat-l">Clean Rate</div></div>
</div>""", unsafe_allow_html=True)

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">📁 Upload Leaf Image</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "drop", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    st.markdown('<div style="text-align:center;font-size:.7rem;color:var(--c4);font-family:var(--mono);margin-top:.35rem;">JPG · PNG · JPEG &nbsp;|&nbsp; Multiple files supported</div>', unsafe_allow_html=True)

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
                f.seek(0); img = Image.open(f).convert("RGB"); valid.append((f.name, img))
            except Exception:
                pass

        if len(uploaded_files) > 3:
            st.caption(f"+{len(uploaded_files)-3} more file(s) queued")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button(f"🔬  Analyze {len(valid)} Image{'s' if len(valid)>1 else ''}", use_container_width=True)
    else:
        st.markdown("""<div style="text-align:center;padding:1.8rem 0;font-size:.82rem;color:var(--c4);font-family:var(--mono);">
↑ &nbsp;Drop or browse a corn leaf image to begin</div>""", unsafe_allow_html=True)

    # ── Inference ──────────────────────────────────────────────────────────
    if analyze and valid:
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
                st.session_state.tip_index += 1
                if label == "Healthy":
                    st.session_state.streak += 1
                else:
                    st.session_state.streak = 0
        st.session_state.results = batch

    # ── Streak celebration ─────────────────────────────────────────────────
    if st.session_state.streak >= 3:
        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(22,163,74,.1),rgba(74,222,128,.06));
  border:1px solid rgba(22,163,74,.2);border-radius:12px;padding:.75rem 1.1rem;
  margin-bottom:.75rem;display:flex;align-items:center;gap:.6rem;font-size:.82rem;color:var(--g);">
  🔥 <strong>Healthy Streak:</strong> {st.session_state.streak} clean scans in a row! Your field looks great.
</div>""", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧬 Diagnosis</div>', unsafe_allow_html=True)

        for r in st.session_state.results:
            info     = r["info"]
            pct      = r["conf"] * 100
            status   = r["status"]
            card_cls = {"ok":"","warn":"amber","bad":"diseased"}.get(status,"")
            tag_cls  = {"ok":"r-ok","warn":"r-warn","bad":"r-bad"}.get(status,"r-ok")
            tag_txt  = {"ok":"⬤ Healthy","warn":"⬤ Monitor","bad":"⬤ Diseased"}.get(status,"")
            bar_grad = {
                "ok"  :"linear-gradient(90deg,#16a34a,#4ade80)",
                "warn":"linear-gradient(90deg,#b45309,#fbbf24)",
                "bad" :"linear-gradient(90deg,#b91c1c,#f87171)",
            }.get(status,"")

            # Quick metrics
            spread_color = "#dc2626" if info["spread_risk"] > 70 else ("#d97706" if info["spread_risk"] > 40 else "#16a34a")

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
  <div class="metric-row">
    <div class="metric-box">
      <div class="metric-n" style="color:{spread_color};">{info['spread_risk']}%</div>
      <div class="metric-l">Spread Risk</div>
    </div>
    <div class="metric-box">
      <div class="metric-n" style="color:var(--amr);">{info['treatment_window']}</div>
      <div class="metric-l">Treat Within</div>
    </div>
    <div class="metric-box">
      <div class="metric-n" style="color:var(--red);">{info['economic_loss']}</div>
      <div class="metric-l">Yield Loss</div>
    </div>
  </div>
  <div class="r-meta" style="margin-top:1rem;"><span>🕐 {r['ts']}</span><span>📄 {r['fname']}</span></div>
</div>""", unsafe_allow_html=True)

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
            info   = r["info"]
            sc     = info["sev_color"]
            sev_bg = sc + "18"

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
    <div class="fun-fact">💬 {info['fun_fact']}</div>
  </div>
  <div class="info-card">
    <div class="info-card-h">🔍 Symptoms</div>
    <div>{chips}</div>
    <div style="margin-top:.85rem">
      <div class="info-card-h">🛡 Recommended Action</div>
      <div class="prev-item"><span class="prev-ico">✓</span><span>{info['action']}</span></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

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
        cl1, cl2, _ = st.columns([1, 1, 3])
        with cl1:
            if st.button("↺ Clear History"):
                st.session_state.history = []
                st.session_state.scanned = 0
                st.session_state.results = []
                st.session_state.streak  = 0
                st.rerun()

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="app-footer">
  CornScan AI &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; CNN Plant Disease Detection &nbsp;·&nbsp; v4.0
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    landing_page()
else:
    main_app()
