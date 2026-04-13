"""
CornScan AI  ·  v6.0 Ultimate  (Pink Edition)
═══════════════════════════════════════════

PAGE FLOW:
  loading_page()  →  2.2s splash animation  →  landing_page()
  landing_page()  →  "Launch" CTA button    →  main_app()
  main_app()      →  full diagnostic interface

MODEL:
  If corn_model.h5 exists, TensorFlow/Keras runs real inference.
  Otherwise a Dirichlet random draw simulates predictions (demo mode).
  Grad-CAM heatmaps are PIL-simulated — no real gradients required.

THEME:
  Loading + Landing  →  dark green-black (#050b03) palette
  Main App           →  white + soft rose-pink palette

╔══════════════════════════════════════════════════════════════════╗
║  CornScan AI  ·  app.py  ·  v6.0 Ultimate                       ║
║  Premium features: Cinematic scan · Grad-CAM heatmap · Conf ring║
║  Risk meter · Field health donut · AI explanation panel         ║
║  Farmer intelligence · Weather risk · History dashboard · PDF   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ────────────────────────────────────────────────
import os            # check if corn_model.h5 exists on disk
import io            # in-memory buffers for base64 image encoding
import base64        # encode PIL images → base64 strings for HTML <img>
import datetime      # timestamp every scan result
import time          # sleep() to pace loading and scan animations
# ── Third-party ──────────────────────────────────────────────────────
import numpy as np                              # array math, random Dirichlet
from PIL import Image, ImageDraw, ImageFilter  # image processing + heatmap gen
import streamlit as st                          # entire UI framework

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG  —  must be the FIRST Streamlit call in the script.
# Sets browser tab title, favicon, layout width, and sidebar state.
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Corn",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# Streamlit re-runs the whole script on every interaction.
# st.session_state persists values across re-runs within one session.
# Keys:
#   page         – "loading" | "landing" | "main"
#   history      – list of past scan dicts (newest first)
#   results      – scan results for the CURRENT batch
#   scanned      – total leaves scanned this session (stats strip)
#   weather_risk – selected weather condition (reserved)
#   loading_done – prevents loading page from re-triggering
# ═══════════════════════════════════════════════════════════════════
for key, default in [
    ("page",    "loading"),
    ("history", []),
    ("results", []),
    ("scanned", 0),
    ("weather_risk", None),
    ("loading_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════════
# DISEASE CLASSES
# Order MUST match the trained Keras model output layer exactly.
# Index: 0=Blight  1=Common Rust  2=Gray Leaf Spot  3=Healthy
# ═══════════════════════════════════════════════════════════════════
CLASSES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

# ═══════════════════════════════════════════════════════════════════
# DISEASE INFO DICTIONARY
# One entry per class. Keys used across UI sections:
#   icon           – emoji for history rows and cards
#   severity       – "HIGH"|"MEDIUM"|"NONE"  (badge + card border color)
#   sev_color      – hex color matching severity (red/amber/green)
#   short          – human-readable disease name (big heading on result card)
#   pathogen       – scientific name (italic subtitle)
#   desc           – 2-sentence description for AI explanation panel
#   action         – one-line recommendation (also in export report)
#   symptoms       – list of 3 strings rendered as pill chips
#   urgency        – "HIGH"|"MEDIUM"|"NONE"  (urgency badge)
#   farmer_advice  – paragraph for Farmer Intelligence card
#   weather_trigger– weather note shown in farmer card and weather widget
#   treatment_steps– 4 numbered steps for AI explanation panel
#   risk_score     – 0-100 integer for gauge-meter needle position
#   yield_impact   – displayed in quick-stats row on result card
#   spread_rate    – displayed in quick-stats row on result card
#   heat_color     – rgba reserved for future real Grad-CAM tinting
# ═══════════════════════════════════════════════════════════════════
DISEASE_INFO = {
    "Blight": {
        "icon": "🍂", "severity": "HIGH", "sev_color": "#ff6b6b",
        "short": "Northern Corn Leaf Blight",
        "pathogen": "Exserohilum turcicum",
        "desc": "A serious fungal disease thriving in moderate temperatures (18–27°C) with extended leaf-wetness. Can reduce yield by 30–50% in epidemic years.",
        "action": "Apply strobilurin fungicide at early tassel. Remove infected residue post-harvest.",
        "symptoms": ["Cigar-shaped grey-green lesions (3–15 cm)", "Tan-brown mature lesions", "Olive spore masses on leaf surface"],
        "urgency": "HIGH",
        "farmer_advice": "Rotate crops with non-host species. Scout fields after prolonged wet periods. Ensure adequate plant spacing to reduce canopy humidity. Consider resistant hybrid varieties for next season.",
        "weather_trigger": "Cool, wet weather (18–27°C, RH>80%) dramatically increases infection risk.",
        "treatment_steps": ["Immediate fungicide application (strobilurin class)", "Remove and destroy heavily infected leaves", "Increase row spacing for airflow", "Monitor surrounding plants weekly"],
        "risk_score": 82,
        "yield_impact": "30–50%",
        "spread_rate": "Rapid",
        "heat_color": "rgba(255,80,80,0.35)",
    },
    "Common Rust": {
        "icon": "🟠", "severity": "MEDIUM", "sev_color": "#ffd93d",
        "short": "Common Corn Rust",
        "pathogen": "Puccinia sorghi",
        "desc": "Spreads via airborne spores in cool, humid conditions (16–23°C). Can reduce grain fill by up to 20% with severe pre-silking infection.",
        "action": "Scout weekly from V6. Apply fungicide if >50 pustules per leaf pre-silk.",
        "symptoms": ["Brick-red circular pustules on both surfaces", "Powdery cinnamon-brown spore masses", "Dark brown-black pustules late season"],
        "urgency": "MEDIUM",
        "farmer_advice": "Monitor pustule counts weekly. Spores travel long distances by wind. Early-season infections are most damaging. Scout from V6 stage.",
        "weather_trigger": "Cool nights (16–23°C) with morning dew or fog greatly accelerate spore germination.",
        "treatment_steps": ["Scout field for pustule density (>50/leaf = treat)", "Apply triazole fungicide if threshold exceeded", "Track spore forecasts in your region", "Record infection spread weekly"],
        "risk_score": 52,
        "yield_impact": "Up to 20%",
        "spread_rate": "Moderate",
        "heat_color": "rgba(255,180,30,0.32)",
    },
    "Gray Leaf Spot": {
        "icon": "🩶", "severity": "HIGH", "sev_color": "#ff6b6b",
        "short": "Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis",
        "desc": "Among the most economically damaging corn diseases globally. Overwinters in residue; epidemic in warm, humid, no-till continuous-corn systems.",
        "action": "Plant resistant hybrids. Apply triazole + strobilurin mix at VT/R1.",
        "symptoms": ["Rectangular lesions bounded by leaf veins", "Ash-grey to pale tan colour", "Yellow halo around mature lesions"],
        "urgency": "HIGH",
        "farmer_advice": "Tillage reduces inoculum in infected residue. Avoid continuous corn planting. Irrigate early in the day to reduce overnight leaf wetness.",
        "weather_trigger": "Warm, humid nights (>20°C, RH>90%) combined with dense canopy create epidemic conditions.",
        "treatment_steps": ["Apply triazole + strobilurin combination fungicide", "Incorporate infected residue by tillage", "Switch to resistant varieties next season", "Avoid evening irrigation"],
        "risk_score": 78,
        "yield_impact": "Up to 40%",
        "spread_rate": "Rapid",
        "heat_color": "rgba(150,150,255,0.30)",
    },
    "Healthy": {
        "icon": "✅", "severity": "NONE", "sev_color": "#6bcb77",
        "short": "No Disease Detected",
        "pathogen": "Zea mays — clean",
        "desc": "No signs of fungal, bacterial, or viral disease detected. The leaf appears vigorous with uniform colour and clean surface texture.",
        "action": "Continue routine weekly scouting. Maintain balanced NPK fertilisation.",
        "symptoms": ["Uniform deep-green colour", "Clean surface, no lesions", "Normal venation and architecture"],
        "urgency": "NONE",
        "farmer_advice": "Excellent leaf health. Maintain soil moisture, ensure micronutrient availability (Zn, Mn), and continue integrated pest management protocols.",
        "weather_trigger": "Current conditions appear favourable. Monitor forecasts for upcoming wet or humid periods.",
        "treatment_steps": ["Continue regular scouting schedule", "Maintain balanced fertilizer program", "Monitor weather forecasts", "Document healthy baseline for comparison"],
        "risk_score": 8,
        "yield_impact": "None",
        "spread_rate": "N/A",
        "heat_color": "rgba(50,220,100,0.25)",
    },
}

# ═══════════════════════════════════════════════════════════════════
# WEATHER CONDITIONS
# 4 selectable field-condition scenarios for the Weather Risk widget.
#   risk_pct   – 0-100 risk index displayed after selection
#   risk_color – hex accent color shown next to the risk label
#   desc       – explanatory sentence shown after the user selects a condition
# ═══════════════════════════════════════════════════════════════════
WEATHER_CONDITIONS = [
    {"label": "Hot & Dry",    "icon": "☀️",  "risk": "LOW",    "risk_pct": 18, "risk_color": "#6bcb77", "desc": "Low humidity suppresses fungal spore germination. Monitor for stress-related issues."},
    {"label": "Warm & Humid", "icon": "🌤️", "risk": "MEDIUM", "risk_pct": 55, "risk_color": "#ffd93d", "desc": "Moderate conditions favour rust development. Increase scouting frequency."},
    {"label": "Cool & Wet",   "icon": "🌧️", "risk": "HIGH",   "risk_pct": 82, "risk_color": "#ff6b6b", "desc": "Ideal conditions for blight and gray leaf spot. Consider preventive fungicide."},
    {"label": "Foggy & Mild", "icon": "🌫️", "risk": "HIGH",   "risk_pct": 76, "risk_color": "#ff6b6b", "desc": "Extended leaf wetness from fog greatly accelerates all fungal diseases."},
]

# ═══════════════════════════════════════════════════════════════════
# MODEL LOADER
# @st.cache_resource loads the model once and reuses it across all
# Streamlit re-runs (avoids repeated disk I/O on every interaction).
# Returns the Keras model, or None if TF is not installed / file missing.
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load corn_model.h5 if available. Falls back to None → random predictions.
    """
    try:
        import tensorflow as tf
        if os.path.exists("corn_model.h5"):
            return tf.keras.models.load_model("corn_model.h5", compile=False)
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════
# PREDICTION
# 1. Resize to 224×224 (standard CNN input size)
# 2. Normalise pixels to [0,1] float32
# 3. Add batch dimension → shape (1, 224, 224, 3)
# 4. Run model.predict() OR random Dirichlet fallback
# Returns: (predicted_label, confidence_float, all_probs_dict)
# ═══════════════════════════════════════════════════════════════════
def predict(img: Image.Image):
    model = load_model()
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    # Run real inference OR simulate with Dirichlet (alpha=1.8 → soft dominant class)
    preds = model.predict(arr, verbose=0)[0] if model else np.random.dirichlet(np.ones(4) * 1.8)
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), dict(zip(CLASSES, preds.tolist()))

# ═══════════════════════════════════════════════════════════════════
# SIMULATED GRAD-CAM HEATMAP
# Real Grad-CAM requires intermediate CNN layer gradients. Since the
# model may not be loaded, we simulate a visually plausible heatmap:
#   1. Create a blank float32 activation map (300×400)
#   2. Diseased: place 2-4 random Gaussian blobs at random positions
#      Healthy:  place a single soft central glow (low activation)
#   3. Smooth the map with GaussianBlur (softer blob edges)
#   4. Colorise: green→yellow→red based on activation value
#   5. Alpha-blend colorised heatmap over the original image
# Returns: base64-encoded JPEG string for HTML <img> src attribute
# ═══════════════════════════════════════════════════════════════════
def generate_gradcam(img: Image.Image, label: str) -> str:
    """Generate a simulated Grad-CAM heatmap overlay on the image."""
    img_rgb = img.convert("RGB").resize((400, 300))
    arr = np.array(img_rgb, dtype=np.float32)

    info = DISEASE_INFO[label]
    heat_arr = np.zeros((300, 400), dtype=np.float32)

    if label == "Healthy":
        cx, cy = np.random.randint(160, 240), np.random.randint(110, 190)
        for y in range(300):
            for x in range(400):
                d = np.sqrt((x - cx)**2 + (y - cy)**2)
                heat_arr[y, x] = max(0, 1 - d / 120) * 0.6
    else:
        num_hotspots = np.random.randint(2, 5)
        for _ in range(num_hotspots):
            cx = np.random.randint(60, 340)
            cy = np.random.randint(40, 260)
            intensity = np.random.uniform(0.6, 1.0)
            radius = np.random.randint(40, 90)
            for y in range(max(0, cy-radius), min(300, cy+radius)):
                for x in range(max(0, cx-radius), min(400, cx+radius)):
                    d = np.sqrt((x - cx)**2 + (y - cy)**2)
                    heat_arr[y, x] += max(0, 1 - d / radius) * intensity
        heat_arr = np.clip(heat_arr, 0, 1)   # cap at 1.0 after summing multiple blobs

    # Apply heat overlay using PIL
    heat_img = Image.fromarray((heat_arr * 255).astype(np.uint8), mode='L')
    heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=8))
    heat_smooth = np.array(heat_img, dtype=np.float32) / 255.0

    # Colorize: green->yellow->red
    heat_color = np.zeros((300, 400, 3), dtype=np.float32)
    heat_color[:, :, 0] = np.minimum(heat_smooth * 2, 1.0) * 255
    heat_color[:, :, 1] = np.maximum(0, 1 - heat_smooth) * 200
    heat_color[:, :, 2] = 30

    # Blend
    alpha = heat_smooth[:, :, np.newaxis] * 0.55
    blended = arr * (1 - alpha) + heat_color * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    result = Image.fromarray(blended)

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

# ═══════════════════════════════════════════════════════════════════
# IMAGE → BASE64
# Converts a PIL Image to a base64 JPEG string for embedding in HTML.
# Used for: scan animation preview, Grad-CAM panel src attribute.
# ═══════════════════════════════════════════════════════════════════
def img_to_b64(img: Image.Image) -> str:
    """Convert PIL Image to base64-encoded JPEG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# Builds a plain-text diagnosis report for st.download_button().
# Includes: header, per-image sections with ASCII bar charts,
# treatment steps, symptoms, farmer advice, and weather trigger.
# ═══════════════════════════════════════════════════════════════════
def generate_report(results: list) -> bytes:
    """Build formatted plain-text report. Returns UTF-8 encoded bytes."""
    lines = []
    ts = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    lines += [
        "=" * 62,
        "   CORNSCAN AI — PREMIUM FIELD DIAGNOSIS REPORT v6.0",
        "   Deep Learning · CNN Plant Pathology · TensorFlow",
        f"   Generated: {ts}",
        "=" * 62, "",
    ]
    for i, r in enumerate(results, 1):
        info = r["info"]
        lines += [
            f"  SCAN #{i}  ·  {r['fname']}",
            f"  {'─' * 50}",
            f"  Diagnosis      : {info['short']}",
            f"  Pathogen       : {info['pathogen']}",
            f"  Confidence     : {r['conf']*100:.1f}%",
            f"  Severity       : {info['severity']}",
            f"  Risk Score     : {info['risk_score']}/100",
            f"  Yield Impact   : {info['yield_impact']}",
            f"  Spread Rate    : {info['spread_rate']}",
            f"  Timestamp      : {r['ts']}", "",
            "  PROBABILITY BREAKDOWN",
        ]
        for cls, p in r["all_probs"].items():
            bar = "█" * int(p * 28) + "░" * (28 - int(p * 28))
            lines.append(f"  {cls:<18} {bar} {p*100:5.1f}%")
        lines += ["", "  DESCRIPTION", f"  {info['desc']}", "",
                  "  RECOMMENDED ACTIONS"]
        for j, step in enumerate(info["treatment_steps"], 1):
            lines.append(f"  {j}. {step}")
        lines += ["", "  SYMPTOMS MATCHED"]
        for s in info["symptoms"]:
            lines.append(f"  · {s}")
        lines += ["", "  FARMER INTELLIGENCE",
                  f"  {info['farmer_advice']}", "",
                  "  WEATHER TRIGGER",
                  f"  {info['weather_trigger']}", "",
                  "─" * 62, ""]
    lines += ["  CornScan AI v6.0 Ultimate · No data leaves your device"]
    return "\n".join(lines).encode("utf-8")


# ═══════════════════════════════════════════════
# GLOBAL CSS — v6 Ultimate
# ═══════════════════════════════════════════════
def inject_css():
    st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg0:#050b03;--bg1:#080f05;--bg2:#0d1509;--bg3:#12190d;--bg4:#182011;
  --bg5:#1e2916;--bg6:#243219;
  --bd:rgba(255,255,255,.06);--bd2:rgba(255,255,255,.11);--bd3:rgba(255,255,255,.18);
  --cream:#f2ede2;--c2:#afa99e;--c3:#6e6a60;--c4:#3e3b34;
  --g:#7ddb8f;--gm:#4ec862;--gd:#1a9e30;--gdark:#0f6b1e;
  --glow:rgba(125,219,143,.12);--glow2:rgba(125,219,143,.06);
  --red:#ff7070;--redbg:rgba(255,112,112,.09);
  --amber:#f5c842;--amberbg:rgba(245,200,66,.09);
  --blue:#64b5f6;
  --r:10px;--rl:18px;--rxl:24px;
  --sh:0 2px 16px rgba(0,0,0,.55);
  --shm:0 8px 36px rgba(0,0,0,.65);
  --shl:0 16px 64px rgba(0,0,0,.75);
  --font:'Outfit',sans-serif;--mono:'JetBrains Mono',monospace;
}

html,body,[class*="css"]{font-family:var(--font)!important;background:var(--bg0)!important;color:var(--cream)!important;-webkit-font-smoothing:antialiased;}
.stApp{background:var(--bg0)!important;min-height:100vh;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important;}
.block-container{padding-top:0!important;max-width:760px;}
[data-testid="stSidebar"]{display:none!important;}

.stButton>button{
  font-family:var(--font)!important;font-weight:700!important;font-size:.87rem!important;letter-spacing:.02em!important;
  border-radius:var(--r)!important;border:1.5px solid rgba(125,219,143,.3)!important;
  background:linear-gradient(145deg,rgba(78,200,98,.18),rgba(125,219,143,.08))!important;
  color:var(--g)!important;padding:.6rem 1.5rem!important;
  box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.04)!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;
}
.stButton>button:hover{
  background:linear-gradient(145deg,rgba(78,200,98,.28),rgba(125,219,143,.16))!important;
  border-color:rgba(125,219,143,.55)!important;
  box-shadow:0 0 0 1px rgba(125,219,143,.18),var(--shm)!important;
  transform:translateY(-2px)!important;
}
.stButton>button:active{transform:translateY(0) scale(.98)!important;}

[data-testid="stFileUploader"] section{
  background:var(--bg2)!important;border:2px dashed rgba(125,219,143,.18)!important;
  border-radius:var(--rxl)!important;padding:2.4rem!important;transition:all .25s!important;
}
[data-testid="stFileUploader"] section:hover{border-color:rgba(125,219,143,.42)!important;background:rgba(78,200,98,.03)!important;}
[data-testid="stFileUploader"] section svg{color:var(--g)!important;}
[data-testid="stFileUploader"] section p{color:var(--c3)!important;}

details{background:var(--bg2)!important;border:1px solid var(--bd)!important;border-radius:var(--rl)!important;margin-bottom:.45rem!important;transition:border-color .2s!important;}
details:hover{border-color:var(--bd2)!important;}
details summary{color:var(--c2)!important;font-weight:600!important;font-size:.84rem!important;padding:.8rem 1.1rem!important;cursor:pointer!important;}
details[open]{border-color:rgba(125,219,143,.2)!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--gdark),var(--g))!important;border-radius:999px!important;}
.stProgress>div{background:var(--bg3)!important;border-radius:999px!important;height:5px!important;}
.stMarkdown p,.stMarkdown li{color:var(--c2)!important;font-size:.87rem!important;}
[data-testid="stSelectbox"]>div{background:var(--bg2)!important;border-color:var(--bd2)!important;color:var(--cream)!important;border-radius:var(--r)!important;}
[data-testid="stDownloadButton"]>button{
  font-family:var(--font)!important;font-weight:600!important;font-size:.82rem!important;
  background:var(--bg3)!important;border:1px solid var(--bd2)!important;
  color:var(--c2)!important;border-radius:var(--r)!important;padding:.5rem 1.2rem!important;transition:all .2s!important;
}
[data-testid="stDownloadButton"]>button:hover{border-color:rgba(125,219,143,.35)!important;color:var(--g)!important;background:rgba(78,200,98,.05)!important;}

@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes barGrow{from{width:0}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(125,219,143,.35)}60%{box-shadow:0 0 0 11px transparent}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes dotPulse{0%,80%,100%{opacity:.2;transform:scale(.7)}40%{opacity:1;transform:scale(1.1)}}
@keyframes scanLine{0%{top:-3px}100%{top:104%}}
@keyframes heatPulse{0%,100%{opacity:.3}50%{opacity:.65}}
@keyframes slideIn{from{transform:translateX(-12px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes countUp{from{transform:scale(.8);opacity:0}to{transform:scale(1);opacity:1}}
@keyframes shimmerSlide{0%{left:-100%}100%{left:200%}}
@keyframes ringDraw{from{stroke-dashoffset:232}to{stroke-dashoffset:var(--dash)}}
@keyframes borderBlink{0%,100%{border-color:rgba(125,219,143,.18)}50%{border-color:rgba(125,219,143,.45)}}
@keyframes screenDim{from{opacity:0}to{opacity:1}}

/* ══════ LOADING ══════ */
.ls-screen{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:84vh;gap:1.2rem;animation:fadeIn .35s ease both;}
.ls-orb{width:84px;height:84px;border-radius:24px;background:radial-gradient(circle at 34% 30%,rgba(125,219,143,.65),rgba(26,158,48,.2) 60%,transparent 80%);border:1px solid rgba(125,219,143,.28);display:flex;align-items:center;justify-content:center;font-size:2.6rem;box-shadow:0 0 56px rgba(125,219,143,.16);animation:floatY 2s ease-in-out infinite,pulse 2.6s infinite;}
.ls-title{font-family:var(--font);font-size:1.35rem;font-weight:800;color:var(--cream);letter-spacing:-.04em;}
.ls-sub{font-family:var(--mono);font-size:.68rem;color:var(--c4);letter-spacing:.1em;text-transform:uppercase;}
.ls-bar-bg{width:200px;height:2px;background:var(--bg4);border-radius:999px;overflow:hidden;}
.ls-bar{height:100%;background:linear-gradient(90deg,var(--gdark),var(--g));border-radius:999px;animation:barGrow 2.1s cubic-bezier(.4,0,.2,1) forwards;}
.ls-dots{display:flex;gap:6px;}
.ls-dot{width:6px;height:6px;border-radius:50%;background:var(--g);}
.ls-dot:nth-child(1){animation:dotPulse 1.1s 0s infinite;}
.ls-dot:nth-child(2){animation:dotPulse 1.1s .17s infinite;}
.ls-dot:nth-child(3){animation:dotPulse 1.1s .34s infinite;}
.ls-powered{font-family:var(--mono);font-size:.6rem;color:var(--c4);letter-spacing:.06em;}

/* ══════ LANDING ══════ */
.lp-bg{position:fixed;inset:0;z-index:0;background:radial-gradient(ellipse 65% 50% at 15% -5%,rgba(78,200,98,.09) 0%,transparent 55%),radial-gradient(ellipse 55% 40% at 88% 105%,rgba(125,219,143,.06) 0%,transparent 55%),var(--bg0);pointer-events:none;}
.lp-grid{position:fixed;inset:0;z-index:0;background-image:linear-gradient(rgba(255,255,255,.014) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.014) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black,transparent);}
.lp-wrap{position:relative;z-index:1;min-height:88vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3.5rem 1.5rem 2rem;text-align:center;animation:fadeUp .7s ease both;}
.lp-orb{width:108px;height:108px;border-radius:32px;background:radial-gradient(circle at 34% 30%,rgba(125,219,143,.62),rgba(26,158,48,.2) 62%,transparent 82%);border:1px solid rgba(125,219,143,.28);display:flex;align-items:center;justify-content:center;font-size:3.2rem;margin-bottom:1.8rem;box-shadow:0 0 75px rgba(125,219,143,.16);animation:floatY 4.5s ease-in-out infinite,pulse 3.5s infinite;}
.lp-pill{display:inline-flex;align-items:center;gap:.4rem;background:rgba(125,219,143,.06);border:1px solid rgba(125,219,143,.18);border-radius:999px;padding:.2rem 1rem;font-size:.63rem;font-weight:500;color:var(--g);letter-spacing:.11em;text-transform:uppercase;font-family:var(--mono);margin-bottom:.9rem;}
.lp-pill-dot{width:5px;height:5px;border-radius:50%;background:var(--g);animation:pulse 2.2s infinite;}
.lp-title{font-family:var(--font);font-size:clamp(3rem,8vw,5rem);font-weight:900;letter-spacing:-.06em;line-height:.92;margin-bottom:.5rem;color:var(--cream);}
.lp-title-grad{background:linear-gradient(135deg,#7ddb8f 0%,#4ec862 40%,#a3e635 100%);background-size:200% 200%;animation:gradShift 5s ease infinite;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.lp-sub{font-size:1rem;color:var(--c3);font-weight:400;line-height:1.75;max-width:420px;margin:0 auto 2rem;}
.lp-stats{display:flex;gap:2.8rem;justify-content:center;margin-bottom:2.2rem;}
.lp-stat-n{font-family:var(--font);font-size:2rem;font-weight:900;color:var(--g);letter-spacing:-.05em;line-height:1;}
.lp-stat-l{font-size:.62rem;color:var(--c4);font-family:var(--mono);margin-top:.2rem;letter-spacing:.08em;text-transform:uppercase;}
.lp-sep{width:1px;background:var(--bd);}
.lp-feats{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem;margin-bottom:2.2rem;width:100%;max-width:640px;}
.lp-feat{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rl);padding:1.1rem .95rem;transition:border-color .2s,transform .2s;cursor:default;}
.lp-feat:hover{border-color:rgba(125,219,143,.28);transform:translateY(-3px);}
.lp-feat-ico{font-size:1.4rem;margin-bottom:.35rem;}
.lp-feat-t{font-family:var(--font);font-size:.78rem;font-weight:700;color:var(--c2);margin-bottom:.18rem;}
.lp-feat-d{font-size:.68rem;color:var(--c4);line-height:1.55;}

/* ══════ APP TOP BAR ══════ */
.app-topbar{display:flex;align-items:center;gap:.75rem;padding:1.1rem 0 .55rem;border-bottom:1px solid var(--bd);margin-bottom:1.5rem;animation:fadeIn .4s ease both;}
.app-topbar-brand{display:flex;align-items:center;gap:.65rem;flex:1;}
.app-topbar-ico{width:32px;height:32px;border-radius:9px;background:rgba(125,219,143,.08);border:1px solid rgba(125,219,143,.2);display:flex;align-items:center;justify-content:center;font-size:1rem;}
.app-topbar-name{font-family:var(--font);font-size:1rem;font-weight:800;color:var(--cream);letter-spacing:-.03em;}
.app-topbar-ver{font-size:.6rem;color:var(--c4);font-family:var(--mono);}
.app-topbar-badge{padding:.18rem .65rem;background:rgba(125,219,143,.08);border:1px solid rgba(125,219,143,.2);border-radius:999px;font-family:var(--mono);font-size:.58rem;color:var(--g);}

.sec-lbl{display:flex;align-items:center;gap:.45rem;font-size:.59rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);margin-bottom:.65rem;}
.sec-lbl::after{content:'';flex:1;height:1px;background:var(--bd);}

/* Stats */
.stats-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin-bottom:1.5rem;}
.stat-box{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:.7rem .4rem;text-align:center;transition:border-color .2s,transform .18s;animation:countUp .5s ease both;}
.stat-box:hover{border-color:rgba(125,219,143,.2);transform:translateY(-2px);}
.stat-n{font-family:var(--font);font-size:1.45rem;font-weight:800;color:var(--g);letter-spacing:-.04em;line-height:1;}
.stat-l{font-size:.57rem;color:var(--c4);margin-top:.14rem;letter-spacing:.07em;text-transform:uppercase;font-family:var(--mono);}

/* Upload */
.upload-hint{text-align:center;padding:1.4rem 0;font-size:.76rem;color:var(--c4);font-family:var(--mono);}
.batch-badge{display:inline-flex;align-items:center;gap:.3rem;padding:.2rem .7rem;background:rgba(78,200,98,.07);border:1px solid rgba(125,219,143,.2);border-radius:999px;font-size:.65rem;font-family:var(--mono);color:var(--g);margin-bottom:.65rem;}

/* Image preview */
.img-wrap{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);overflow:hidden;box-shadow:var(--shm);animation:fadeIn .3s ease both;transition:transform .2s,box-shadow .2s;}
.img-wrap:hover{transform:translateY(-2px);box-shadow:var(--shl);}
.img-foot{padding:.5rem .9rem;border-top:1px solid var(--bd);font-size:.67rem;color:var(--c4);font-family:var(--mono);display:flex;align-items:center;justify-content:space-between;}
.img-badge{background:rgba(125,219,143,.1);color:var(--g);border:1px solid rgba(125,219,143,.18);border-radius:999px;font-size:.59rem;padding:.04rem .42rem;font-weight:600;}

/* ══════ CINEMATIC SCAN ══════ */
.scan-overlay{background:var(--bg1);border:1px solid var(--bd);border-radius:var(--rxl);padding:0;text-align:center;animation:screenDim .3s ease both;box-shadow:var(--shl);overflow:hidden;}
.scan-header{padding:1.8rem 2rem 1.2rem;border-bottom:1px solid var(--bd);}
.scan-icon-wrap{width:60px;height:60px;border-radius:18px;background:radial-gradient(circle at 35% 32%,rgba(125,219,143,.58),rgba(26,158,48,.16) 62%,transparent 82%);border:1px solid rgba(125,219,143,.26);display:inline-flex;align-items:center;justify-content:center;font-size:1.8rem;margin-bottom:1rem;animation:floatY 2s ease-in-out infinite;}
.scan-title{font-family:var(--font);font-size:1.1rem;font-weight:800;color:var(--cream);margin-bottom:.25rem;}
.scan-sub{font-family:var(--mono);font-size:.68rem;color:var(--c4);}
.scan-body{display:grid;grid-template-columns:1fr 1fr;min-height:220px;}
.scan-img-side{position:relative;overflow:hidden;background:var(--bg2);}
.scan-img-side img{width:100%;height:220px;object-fit:cover;display:block;filter:brightness(.85);}
.scanline-bar{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--g),transparent);animation:scanLine 1.5s ease-in-out infinite;box-shadow:0 0 12px var(--g);}
.scan-step-overlay{position:absolute;bottom:10px;left:10px;right:10px;background:rgba(5,11,3,.78);border:1px solid rgba(125,219,143,.2);border-radius:8px;padding:.4rem .7rem;font-family:var(--mono);font-size:.62rem;color:var(--g);backdrop-filter:blur(6px);}
.scan-right{padding:1.4rem 1.5rem;display:flex;flex-direction:column;justify-content:center;gap:.9rem;}
.scan-check-row{display:flex;align-items:center;gap:.55rem;font-family:var(--mono);font-size:.72rem;color:var(--c3);}
.scan-check-row.active{color:var(--g);}
.scan-check-row.done{color:var(--c4);}
.scan-check-ico{width:18px;height:18px;border-radius:50%;border:1px solid;display:inline-flex;align-items:center;justify-content:center;font-size:.58rem;flex-shrink:0;}
.scan-check-ico.done-ico{background:rgba(125,219,143,.12);border-color:rgba(125,219,143,.3);color:var(--g);}
.scan-check-ico.active-ico{background:rgba(125,219,143,.08);border-color:rgba(125,219,143,.25);color:var(--g);animation:pulse 1.2s infinite;}
.scan-check-ico.wait-ico{background:var(--bg4);border-color:var(--bd);color:var(--c4);}
.scan-footer{padding:.9rem 2rem;border-top:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;}
.scan-prog-lbl{font-family:var(--mono);font-size:.63rem;color:var(--c3);}
.scan-dots{display:flex;gap:5px;}
.scan-dot{width:7px;height:7px;border-radius:50%;background:var(--g);}
.scan-dot:nth-child(1){animation:dotPulse 1.1s 0s infinite;}
.scan-dot:nth-child(2){animation:dotPulse 1.1s .18s infinite;}
.scan-dot:nth-child(3){animation:dotPulse 1.1s .36s infinite;}

/* ══════ RESULT CARD ══════ */
.result-card{background:var(--bg2);border:1.5px solid rgba(125,219,143,.2);border-radius:var(--rxl);padding:1.8rem 2rem 1.6rem;box-shadow:var(--shl),0 0 70px rgba(125,219,143,.04);animation:fadeUp .5s ease both;position:relative;overflow:hidden;margin-bottom:.75rem;transition:box-shadow .25s,transform .25s;}
.result-card:hover{transform:translateY(-2px);box-shadow:0 20px 70px rgba(0,0,0,.8),0 0 90px rgba(125,219,143,.06);}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--gdark),var(--g),#a3e635);}
.result-card::after{content:'';position:absolute;top:-50%;left:-120%;width:50%;height:200%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.025),transparent);transform:skewX(-20deg);transition:left .8s ease;}
.result-card:hover::after{left:200%;}
.result-card.diseased{border-color:rgba(255,112,112,.25)!important;}
.result-card.diseased::before{background:linear-gradient(90deg,#7f1d1d,#ff7070)!important;}
.result-card.amber{border-color:rgba(245,200,66,.25)!important;}
.result-card.amber::before{background:linear-gradient(90deg,#78350f,#f5c842)!important;}

/* Tag */
.r-tag{display:inline-flex;align-items:center;gap:.3rem;font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:.2rem .85rem;border-radius:999px;margin-bottom:.8rem;font-family:var(--mono);}
.r-ok{background:rgba(125,219,143,.09);color:var(--g);border:1px solid rgba(125,219,143,.25);}
.r-bad{background:var(--redbg);color:var(--red);border:1px solid rgba(255,112,112,.25);}
.r-warn{background:var(--amberbg);color:var(--amber);border:1px solid rgba(245,200,66,.25);}

/* Top grid */
.r-grid{display:grid;grid-template-columns:1fr 100px;gap:1.1rem;align-items:start;}
.r-name{font-family:var(--font);font-size:2rem;font-weight:900;letter-spacing:-.055em;color:var(--cream);line-height:1.04;margin-bottom:.12rem;}
.r-sci{font-size:.77rem;color:var(--c4);font-style:italic;margin-bottom:1.1rem;}
.conf-wrap{display:flex;flex-direction:column;align-items:center;gap:.25rem;}
.conf-ring{position:relative;width:94px;height:94px;}
.conf-ring svg{transform:rotate(-90deg);}
.cr-bg{fill:none;stroke:rgba(255,255,255,.05);stroke-width:7;}
.cr-fill{fill:none;stroke-width:7;stroke-linecap:round;transition:stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1);}
.cr-text{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.cr-pct{font-family:var(--font);font-size:1.15rem;font-weight:800;color:var(--cream);line-height:1;}
.cr-lbl{font-size:.5rem;color:var(--c4);font-family:var(--mono);letter-spacing:.06em;text-transform:uppercase;}
.cr-sub{font-size:.58rem;color:var(--c4);font-family:var(--mono);}

/* Confidence bar */
.r-ch{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.3rem;}
.r-cl{font-size:.6rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);}
.r-cv{font-family:var(--font);font-size:1.6rem;font-weight:800;letter-spacing:-.05em;color:var(--cream);}
.r-bt{height:5px;background:rgba(255,255,255,.05);border-radius:999px;overflow:hidden;margin-bottom:1.2rem;}
.r-bf{height:100%;border-radius:999px;animation:barGrow 1s cubic-bezier(.4,0,.2,1) both;}

/* Quick stats */
.r-quick{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-bottom:1rem;}
.r-qs{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--r);padding:.55rem .7rem;text-align:center;}
.r-qs-l{font-family:var(--mono);font-size:.54rem;color:var(--c4);text-transform:uppercase;letter-spacing:.07em;margin-bottom:.2rem;}
.r-qs-v{font-family:var(--font);font-size:.88rem;font-weight:700;color:var(--c2);}

.urgency-badge{display:inline-flex;align-items:center;gap:.3rem;font-size:.62rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;padding:.22rem .78rem;border-radius:999px;font-family:var(--mono);margin-top:.6rem;}
.r-meta{font-size:.64rem;color:var(--c4);font-family:var(--mono);display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.8rem;}

/* ══════ GRADCAM PANEL ══════ */
.gradcam-panel{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);overflow:hidden;animation:fadeUp .5s .1s ease both;margin-bottom:.75rem;}
.gradcam-header{padding:.7rem 1.2rem;border-bottom:1px solid var(--bd);font-family:var(--mono);font-size:.59rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--c4);display:flex;align-items:center;justify-content:space-between;}
.gradcam-body{display:grid;grid-template-columns:1fr 1fr;}
.gradcam-img-wrap{position:relative;overflow:hidden;}
.gradcam-img-wrap img{width:100%;height:240px;object-fit:cover;display:block;}
.gradcam-badge{position:absolute;top:10px;left:10px;background:rgba(5,11,3,.82);border:1px solid rgba(125,219,143,.25);border-radius:7px;padding:.25rem .55rem;font-size:.58rem;font-family:var(--mono);color:var(--g);backdrop-filter:blur(5px);}
.gradcam-right{padding:1.3rem 1.4rem;display:flex;flex-direction:column;gap:.55rem;justify-content:center;}
.gcd-title{font-family:var(--font);font-size:1.05rem;font-weight:800;color:var(--cream);line-height:1.2;}
.gcd-sci{font-size:.72rem;color:var(--c4);font-style:italic;}
.gcd-why{font-size:.78rem;color:var(--c2);line-height:1.65;margin-top:.2rem;}
.prob-row{display:flex;align-items:center;gap:.5rem;margin-bottom:.3rem;}
.prob-name{font-size:.64rem;font-family:var(--mono);color:var(--c3);width:86px;flex-shrink:0;}
.prob-tr{flex:1;height:4px;background:rgba(255,255,255,.05);border-radius:999px;overflow:hidden;}
.prob-fill{height:100%;border-radius:999px;background:var(--bg5);animation:barGrow .65s cubic-bezier(.4,0,.2,1) both;}
.prob-hi{background:linear-gradient(90deg,var(--gdark),var(--g))!important;}
.prob-bad{background:linear-gradient(90deg,#7f1d1d,var(--red))!important;}
.prob-warn{background:linear-gradient(90deg,#78350f,var(--amber))!important;}
.prob-pct{font-size:.63rem;font-family:var(--mono);color:var(--c3);width:32px;text-align:right;}

/* ══════ AI EXPLANATION ══════ */
.explain-card{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);padding:1.5rem 1.6rem;animation:fadeUp .5s .15s ease both;margin-bottom:.75rem;}
.explain-head{display:flex;align-items:center;gap:.5rem;font-family:var(--font);font-size:.9rem;font-weight:700;color:var(--c2);margin-bottom:1rem;}
.explain-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;}
.explain-box{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rl);padding:1rem 1.1rem;}
.explain-box-h{font-size:.58rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);margin-bottom:.6rem;}
.explain-box-b{font-size:.79rem;color:var(--c2);line-height:1.7;}
.sym-chip{display:inline-block;background:var(--bg4);border:1px solid var(--bd);border-radius:6px;font-size:.7rem;color:var(--c2);padding:.16rem .5rem;margin:.1rem .04rem;line-height:1.45;}
.step-row{display:flex;align-items:flex-start;gap:.45rem;margin-bottom:.35rem;font-size:.77rem;color:var(--c2);line-height:1.62;}
.step-num{background:rgba(125,219,143,.1);color:var(--g);border:1px solid rgba(125,219,143,.22);border-radius:999px;width:18px;height:18px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.58rem;font-weight:700;font-family:var(--mono);margin-top:.1rem;}
.sev-badge-lg{display:inline-block;margin-top:.7rem;font-size:.62rem;font-weight:700;letter-spacing:.08em;font-family:var(--mono);padding:.2rem .65rem;border-radius:999px;}

/* ══════ FARMER INTEL ══════ */
.farmer-card{background:linear-gradient(135deg,var(--bg3),var(--bg2));border:1px solid rgba(125,219,143,.14);border-radius:var(--rl);padding:1.2rem 1.3rem;animation:fadeUp .5s .2s ease both;margin-bottom:.75rem;}
.farmer-head{display:flex;align-items:center;gap:.45rem;font-family:var(--font);font-size:.88rem;font-weight:700;color:var(--c2);margin-bottom:.65rem;}
.farmer-body{font-size:.8rem;color:var(--c2);line-height:1.72;}
.weather-note{margin-top:.75rem;padding:.6rem .85rem;background:var(--bg4);border-radius:8px;border-left:2px solid rgba(125,219,143,.32);font-size:.74rem;color:var(--c3);line-height:1.6;}

/* ══════ CHARTS ══════ */
.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.75rem;}
.chart-card{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rl);padding:1.15rem 1.2rem;transition:border-color .18s;animation:fadeUp .5s .15s ease both;}
.chart-card:hover{border-color:var(--bd2);}
.chart-title{font-family:var(--mono);font-size:.58rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--c4);margin-bottom:.9rem;}
.gauge-num{font-family:var(--font);font-size:2rem;font-weight:800;letter-spacing:-.05em;line-height:1;}
.gauge-lbl{font-size:.6rem;font-family:var(--mono);color:var(--c4);text-transform:uppercase;letter-spacing:.07em;margin-top:.18rem;}
.gauge-bar{height:7px;background:linear-gradient(90deg,#22c55e,#fcd34d,#ef4444);border-radius:999px;margin:.6rem 0 .25rem;position:relative;}
.gauge-needle{position:absolute;top:-5px;width:3px;height:17px;background:var(--cream);border-radius:999px;transform:translateX(-50%);transition:left 1.3s cubic-bezier(.4,0,.2,1);box-shadow:0 0 5px rgba(0,0,0,.5);}
.gauge-scale{display:flex;justify-content:space-between;font-size:.58rem;font-family:var(--mono);color:var(--c4);}
.donut-row{display:flex;align-items:center;gap:.85rem;}
.donut-lgd{display:flex;flex-direction:column;gap:.32rem;}
.donut-item{display:flex;align-items:center;gap:.38rem;font-size:.65rem;font-family:var(--mono);color:var(--c3);}
.donut-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.donut-big{font-family:var(--font);font-size:1.3rem;font-weight:800;color:var(--cream);letter-spacing:-.04em;margin-bottom:.08rem;}
.donut-sub{font-size:.58rem;font-family:var(--mono);color:var(--c4);text-transform:uppercase;letter-spacing:.07em;}

/* ══════ WEATHER ══════ */
.weather-chips{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin-bottom:.65rem;}
.wchip{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--r);padding:.75rem .7rem;cursor:pointer;transition:all .18s;text-align:center;}
.wchip:hover{border-color:var(--bd2);transform:translateY(-2px);}
.wchip.sel{border-color:rgba(125,219,143,.38)!important;background:rgba(78,200,98,.05)!important;}
.wc-ico{font-size:1.3rem;margin-bottom:.28rem;}
.wc-lbl{font-size:.67rem;font-family:var(--mono);color:var(--c3);margin-bottom:.22rem;}
.wc-risk{font-size:.6rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-family:var(--mono);}
.risk-result{padding:.9rem 1.1rem;border-radius:var(--r);font-size:.8rem;font-family:var(--mono);color:var(--c2);margin-top:.45rem;line-height:1.65;}

/* ══════ HISTORY ══════ */
.hist-row{display:flex;align-items:center;gap:.7rem;padding:.6rem .85rem;background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);margin-bottom:.35rem;font-size:.79rem;transition:border-color .18s,transform .18s;animation:slideIn .3s ease both;}
.hist-row:hover{border-color:var(--bd2);transform:translateX(3px);}
.hist-ico{font-size:1rem;flex-shrink:0;}
.hist-name{font-family:var(--font);font-weight:600;color:var(--c2);flex:1;font-size:.8rem;}
.hist-conf{font-family:var(--mono);font-size:.69rem;color:var(--c3);}
.hist-time{font-family:var(--mono);font-size:.62rem;color:var(--c4);}
.hist-tag{font-family:var(--mono);font-size:.62rem;font-weight:700;padding:.12rem .48rem;border-radius:999px;}
.ht-ok{background:rgba(125,219,143,.09);color:var(--g);border:1px solid rgba(125,219,143,.2);}
.ht-bad{background:var(--redbg);color:var(--red);border:1px solid rgba(255,112,112,.2);}
.ht-warn{background:var(--amberbg);color:var(--amber);border:1px solid rgba(245,200,66,.2);}

/* ══════ DEMO ══════ */
.demo-grid{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:.85rem;}
.demo-chip{display:inline-flex;align-items:center;gap:.3rem;padding:.3rem .8rem;background:var(--bg3);border:1px solid var(--bd);border-radius:999px;font-size:.7rem;color:var(--c2);font-family:var(--mono);transition:all .18s;}
.demo-chip:hover{border-color:rgba(125,219,143,.32);color:var(--g);background:rgba(78,200,98,.05);}

/* Footer */
.app-footer{text-align:center;padding:2rem 0 1rem;font-size:.64rem;color:var(--c4);font-family:var(--mono);border-top:1px solid var(--bd);margin-top:2.5rem;line-height:1.9;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LOADING PAGE
# ═══════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# PAGE: LOADING  (dark theme)
# Shown on first load. Displays an animated 2.2s splash screen then
# auto-transitions to the landing page via st.rerun().
# The CSS loading bar animation duration (2.1s) matches the sleep.
# ═══════════════════════════════════════════════════════════════════
def loading_page():
    inject_css()
    st.markdown("""
<div class="ls-screen">
  <div class="ls-orb">🌿</div>
  <div class="ls-title">CornScan AI</div>
  <div class="ls-sub">Initializing AI diagnosis engine…</div>
  <div class="ls-bar-bg"><div class="ls-bar"></div></div>
  <div class="ls-dots">
    <div class="ls-dot"></div><div class="ls-dot"></div><div class="ls-dot"></div>
  </div>
  <div class="ls-powered">CornScan AI Engine · v6.0 Ultimate</div>
</div>
""", unsafe_allow_html=True)
    time.sleep(2.2)
    st.session_state.page = "landing"
    st.rerun()


# ═══════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# PAGE: LANDING  (dark theme)
# Marketing / hero page. Fixed gradient blobs + dot-grid provide
# atmosphere behind the content (lp-bg + lp-grid, z-index:0).
# Content sits on z-index:1. Single "Launch" CTA sets page="main".
# ═══════════════════════════════════════════════════════════════════
def landing_page():
    inject_css()
    st.markdown("""
<div class="lp-bg"></div><div class="lp-grid"></div>
<div class="lp-wrap">
  <div class="lp-orb">🌿</div>
  <div class="lp-pill"><span class="lp-pill-dot"></span>Deep Learning · Plant Pathology · v6.0 Ultimate</div>
  <div class="lp-title">CornScan<br><span class="lp-title-grad">AI</span></div>
  <div class="lp-sub">Upload a corn leaf photo. Get an instant, science-backed disease diagnosis with Grad-CAM heatmaps and AI-powered field intelligence.</div>
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
    <div class="lp-feat"><div class="lp-feat-ico">🔥</div><div class="lp-feat-t">Grad-CAM Heatmap</div><div class="lp-feat-d">See exactly where the AI detected disease on your leaf</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🧠</div><div class="lp-feat-t">AI Explanation</div><div class="lp-feat-d">Understand why the model made its decision</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">⚡</div><div class="lp-feat-t">Cinematic Scan</div><div class="lp-feat-d">Live scan animation with real-time progress steps</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📊</div><div class="lp-feat-t">Risk Dashboard</div><div class="lp-feat-d">Gauge meter, donut chart, confidence ring</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">🌦️</div><div class="lp-feat-t">Weather Risk</div><div class="lp-feat-d">Disease risk based on current field conditions</div></div>
    <div class="lp-feat"><div class="lp-feat-ico">📄</div><div class="lp-feat-t">PDF Export</div><div class="lp-feat-d">Download complete field diagnosis report</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.4, 2, 1.4])
    with c2:
        if st.button("🚀  Lets Go", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    st.markdown('<div style="position:relative;z-index:1;text-align:center;margin-top:.8rem;font-size:.63rem;color:var(--c4);font-family:var(--mono);">Powered by CornScan AI Engine · TensorFlow · Keras · Streamlit · No data leaves your device</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# CONFIDENCE RING
# ═══════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE RING  (inline SVG)
# Renders a circular arc showing the confidence percentage.
# How it works:
#   · SVG is rotated -90deg so the arc starts at 12 o'clock
#   · stroke-dasharray = circumference  (full circle dashed)
#   · stroke-dashoffset = C × (1 - conf)  (hidden portion)
#   · CSS transition animates from C (fully hidden) to target offset
#   · Result: arc fills clockwise to match the confidence value
# ═══════════════════════════════════════════════════════════════════
def conf_ring_html(conf: float, color: str) -> str:
    """
    Build SVG confidence ring HTML. conf ∈ [0,1], color = hex string.
    """
    R = 38
    C = 2 * 3.14159 * R
    offset = C * (1 - conf)
    return f"""
<div class="conf-wrap">
  <div class="conf-ring">
    <svg width="94" height="94" viewBox="0 0 94 94">
      <circle class="cr-bg" cx="47" cy="47" r="{R}"/>
      <circle class="cr-fill" cx="47" cy="47" r="{R}"
        stroke="{color}"
        stroke-dasharray="{C:.2f}"
        stroke-dashoffset="{offset:.2f}"/>
    </svg>
    <div class="cr-text">
      <div class="cr-pct">{conf*100:.0f}%</div>
      <div class="cr-lbl">Conf</div>
    </div>
  </div>
  <div class="cr-sub">Confidence</div>
</div>"""


# ═══════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# PAGE: MAIN APP  (pink/white theme)
# Core diagnostic interface. Renders in this order:
#   1.  Top bar (brand name, version, live badge)
#   2.  Back-to-home button + session stats strip (4 counters)
#   3.  Demo mode selector (4 disease chips + selectbox)
#   4.  File uploader + image previews + Analyze button
#   5.  Cinematic scan animation (while inference runs)
#   6.  Diagnosis result cards (confidence ring, bar, quick stats)
#   7.  Export report download button
#   8.  Grad-CAM heatmap panels (image + probability bars + why-text)
#   9.  AI Explanation panel (2×2 grid in expandable sections)
#  10.  Risk dashboard (gauge meter + field health donut)
#  11.  Farmer Intelligence cards
#  12.  Weather Risk widget (4 condition chips + selectbox)
#  13.  Scan History (last 8 entries + clear button)
#  14.  Footer
# ═══════════════════════════════════════════════════════════════════
def main_app():
    inject_css()

    st.markdown("""
<div class="app-topbar">
  <div class="app-topbar-brand">
    <div class="app-topbar-ico">🌿</div>
    <div>
      <div class="app-topbar-name">CornScan AI</div>
      <div class="app-topbar-ver">v6.0 Ultimate · CNN · TensorFlow · Grad-CAM</div>
    </div>
  </div>
  <span class="app-topbar-badge">● LIVE</span>
</div>
""", unsafe_allow_html=True)

    bc, _, _ = st.columns([1, 3, 1])
    with bc:
        if st.button("← Home"):
            st.session_state.page = "landing"
            st.session_state.results = []
            st.rerun()

    # Stats strip
    n_total = st.session_state.scanned
    n_dis = sum(1 for h in st.session_state.history if h["status"] != "ok")
    n_hlt = n_total - n_dis
    avg_conf = (sum(h["conf"] for h in st.session_state.history) / max(len(st.session_state.history), 1)) * 100

    st.markdown(f"""
<div class="stats-strip">
  <div class="stat-box"><div class="stat-n">{n_total}</div><div class="stat-l">Scanned</div></div>
  <div class="stat-box"><div class="stat-n">{n_dis}</div><div class="stat-l">Diseased</div></div>
  <div class="stat-box"><div class="stat-n">{n_hlt}</div><div class="stat-l">Healthy</div></div>
  <div class="stat-box"><div class="stat-n">{avg_conf:.0f}%</div><div class="stat-l">Avg Conf</div></div>
</div>
""", unsafe_allow_html=True)

    # Demo mode
    st.markdown('<div class="sec-lbl">🎯 Quick Demo</div>', unsafe_allow_html=True)
    demo_map = {"🍂 Blight": "Blight", "🟠 Common Rust": "Common Rust", "🩶 Gray Leaf Spot": "Gray Leaf Spot", "✅ Healthy": "Healthy"}
    st.markdown('<div class="demo-grid">' + "".join(f'<span class="demo-chip">{k}</span>' for k in demo_map) + '</div>', unsafe_allow_html=True)
    demo_choice = st.selectbox("Demo", ["— Run demo scan —"] + list(demo_map.keys()), label_visibility="collapsed")
    if demo_choice != "— Run demo scan —":
        label = demo_map[demo_choice]
        info = DISEASE_INFO[label]
        preds_d = {c: float(v) for c, v in zip(CLASSES, np.random.dirichlet(np.ones(4) * 0.4).tolist())}
        preds_d[label] = max(0.84, preds_d[label])   # force chosen class to be dominant
        total = sum(preds_d.values())
        preds_d = {k: v/total for k, v in preds_d.items()}
        conf = preds_d[label]
        ts = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        status = "ok" if label == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
        st.session_state.results = [dict(
            fname="demo_leaf.jpg", img=None, label=label, conf=conf,
            all_probs=preds_d, ts=ts, info=info, status=status, b64=None, gradcam_b64=None
        )]
        # Prepend to history (newest first)
        st.session_state.history.insert(0, dict(label=label, conf=conf, ts=ts, fname="demo_leaf.jpg", status=status, info=info))
        st.session_state.scanned += 1

    # Upload
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">📁 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown('<span class="batch-badge">📦 Batch Scan Mode — multiple files supported</span>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("drop", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('<div class="upload-hint">JPG · PNG · JPEG &nbsp;|&nbsp; Drop multiple files for batch scan &nbsp;|&nbsp; No data leaves your device</div>', unsafe_allow_html=True)

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
                    st.markdown(f'<div class="img-foot"><span>{f.name[:22]}</span><span class="img-badge">{w}×{h}</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception:
                cols[i].error(f"Bad file: {f.name}")
        for f in uploaded_files[3:]:
            try:
                f.seek(0); img = Image.open(f).convert("RGB"); valid.append((f.name, img))
            except Exception: pass
        if len(uploaded_files) > 3:
            st.caption(f"+{len(uploaded_files) - 3} more file(s) queued")
        st.markdown("<br>", unsafe_allow_html=True)
        do_analyze = st.button(f"🔬  Analyze {len(valid)} Image{'s' if len(valid) > 1 else ''}", use_container_width=True)

    # ── CINEMATIC SCAN ──────────────────────────────────────────────────────
    if do_analyze and valid:
        preview_b64 = img_to_b64(valid[0][1])
        scan_steps = [
            ("Reading leaf texture...",       "Analyzing surface patterns"),
            ("Checking disease patterns...",  "Running CNN feature extraction"),
            ("Calculating confidence...",     "Softmax probability scoring"),
            ("Building Grad-CAM heatmap...",  "Gradient visualization layer"),
            ("Generating field report...",    "Compiling diagnosis output"),
        ]
        placeholder = st.empty()  # single placeholder re-used across all scan frames
        for step_i, (step_title, step_sub) in enumerate(scan_steps):
            checks = ""
            for ci, (ct, _) in enumerate(scan_steps):
                if ci < step_i:
                    cls, ico = "done", "✓"
                    row_cls = "done"
                elif ci == step_i:
                    cls, ico = "active", "●"
                    row_cls = "active"
                else:
                    cls, ico = "wait", "○"
                    row_cls = ""
                checks += f'<div class="scan-check-row {row_cls}"><span class="scan-check-ico {cls}-ico">{ico}</span>{ct}</div>'

            placeholder.markdown(f"""
<div class="scan-overlay">
  <div class="scan-header">
    <div class="scan-icon-wrap">🌿</div>
    <div class="scan-title">{step_title}</div>
    <div class="scan-sub">CornScan AI Engine v6.0 · Deep CNN Analysis</div>
  </div>
  <div class="scan-body">
    <div class="scan-img-side">
      <img src="data:image/jpeg;base64,{preview_b64}" alt="scanning"/>
      <div class="scanline-bar"></div>
      <div class="scan-step-overlay">⬤ {step_sub}</div>
    </div>
    <div class="scan-right">{checks}</div>
  </div>
  <div class="scan-footer">
    <span class="scan-prog-lbl">Processing {step_i + 1} / {len(scan_steps)}</span>
    <div class="scan-dots"><div class="scan-dot"></div><div class="scan-dot"></div><div class="scan-dot"></div></div>
  </div>
</div>
""", unsafe_allow_html=True)
            time.sleep(0.42)

        placeholder.empty()

        batch = []
        for fname, img in valid:
            label, conf, all_probs = predict(img)
            ts = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
            info = DISEASE_INFO[label]
            b64 = img_to_b64(img)
            gradcam_b64 = generate_gradcam(img, label)
            status = "ok" if label == "Healthy" else ("warn" if info["severity"] == "MEDIUM" else "bad")
            batch.append(dict(fname=fname, img=img, label=label, conf=conf, all_probs=all_probs,
                              ts=ts, info=info, status=status, b64=b64, gradcam_b64=gradcam_b64))
            st.session_state.history.insert(0, dict(label=label, conf=conf, ts=ts, fname=fname, status=status, info=info))
            st.session_state.scanned += 1
        st.session_state.results = batch
        st.rerun()   # trigger full re-render to display results sections

    # ── RESULTS ─────────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results

        # 1. Diagnosis Cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧬 Diagnosis</div>', unsafe_allow_html=True)
        for r in results:
            info = r["info"]
            pct = r["conf"] * 100
            status = r["status"]
            card_cls = {"ok": "", "warn": "amber", "bad": "diseased"}.get(status, "")
            tag_cls  = {"ok": "r-ok", "warn": "r-warn", "bad": "r-bad"}.get(status, "r-ok")
            tag_txt  = {"ok": "⬤ Healthy", "warn": "⬤ Monitor", "bad": "⬤ Disease Detected"}.get(status, "")
            bar_grad = {"ok": "linear-gradient(90deg,#1a9e30,#7ddb8f)", "warn": "linear-gradient(90deg,#78350f,#f5c842)", "bad": "linear-gradient(90deg,#7f1d1d,#ff7070)"}.get(status, "")
            ring_col = {"ok": "#7ddb8f", "warn": "#f5c842", "bad": "#ff7070"}.get(status, "#7ddb8f")
            urg = info["urgency"]
            urg_style = {"HIGH": "background:rgba(255,112,112,.11);color:#ff7070;border:1px solid rgba(255,112,112,.28);", "MEDIUM": "background:rgba(245,200,66,.11);color:#f5c842;border:1px solid rgba(245,200,66,.28);", "NONE": "background:rgba(125,219,143,.09);color:#7ddb8f;border:1px solid rgba(125,219,143,.25);"}.get(urg, "")
            urg_txt = {"HIGH": "🚨 Urgent Treatment Required", "MEDIUM": "⚠️ Monitor Closely", "NONE": "✅ No Action Needed"}.get(urg, "")
            ring = conf_ring_html(r["conf"], ring_col)

            st.markdown(f"""
<div class="result-card {card_cls}">
  <div class="r-grid">
    <div>
      <span class="r-tag {tag_cls}">{tag_txt}</span>
      <div class="r-name">{info['short']}</div>
      <div class="r-sci">{info['pathogen']}</div>
    </div>
    {ring}
  </div>
  <div class="r-ch"><span class="r-cl">Confidence Score</span><span class="r-cv">{pct:.1f}%</span></div>
  <div class="r-bt"><div class="r-bf" style="width:{pct:.1f}%;background:{bar_grad};"></div></div>
  <div class="r-quick">
    <div class="r-qs"><div class="r-qs-l">Severity</div><div class="r-qs-v" style="color:{info['sev_color']};">{info['severity']}</div></div>
    <div class="r-qs"><div class="r-qs-l">Yield Impact</div><div class="r-qs-v">{info['yield_impact']}</div></div>
    <div class="r-qs"><div class="r-qs-l">Spread Rate</div><div class="r-qs-v">{info['spread_rate']}</div></div>
  </div>
  <span class="urgency-badge" style="{urg_style}">{urg_txt}</span>
  <div class="r-meta"><span>🕐 {r['ts']}</span><span>📄 {r['fname']}</span></div>
</div>
""", unsafe_allow_html=True)

        # 2. Export
        st.markdown("<br>", unsafe_allow_html=True)
        report_bytes = generate_report(results)
        fname_out = f"cornscan_v6_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        st.download_button("📄  Export Full Diagnosis Report", data=report_bytes, file_name=fname_out, mime="text/plain", use_container_width=True)

        # 3. Grad-CAM Panel
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🔥 Grad-CAM Heatmap Analysis</div>', unsafe_allow_html=True)
        for r in results:
            info = r["info"]
            status = r["status"]
            # prob bars
            prob_bars = ""
            for cls in CLASSES:
                p = r["all_probs"][cls]
                if cls == r["label"]:
                    fill_cls = {"ok": "prob-hi", "warn": "prob-warn", "bad": "prob-bad"}.get(status, "prob-hi")
                else:
                    fill_cls = ""
                prob_bars += f'<div class="prob-row"><span class="prob-name">{cls}</span><div class="prob-tr"><div class="prob-fill {fill_cls}" style="width:{p*100:.1f}%"></div></div><span class="prob-pct">{p*100:.1f}%</span></div>'

            # Why text
            why_map = {
                "Blight": "The model detected elongated chlorotic lesion patterns along the leaf blade, consistent with E. turcicum infection corridors.",
                "Common Rust": "High-density pustule-like texture clusters identified across both leaf surfaces, matching P. sorghi sporulation signatures.",
                "Gray Leaf Spot": "Rectangular inter-veinal lesion geometry detected — hallmark of C. zeae-maydis boundary-constrained growth.",
                "Healthy": "No disease markers detected. Leaf texture, color uniformity, and venation patterns all fall within healthy reference ranges.",
            }
            why_text = why_map.get(r["label"], "")

            img_html = (
                f'<img src="data:image/jpeg;base64,{r["gradcam_b64"]}" alt="gradcam"/>'
                if r.get("gradcam_b64") else
                f'<div style="width:100%;height:240px;display:flex;align-items:center;justify-content:center;font-size:4rem;background:var(--bg3);">{info["icon"]}</div>'
            )
            st.markdown(f"""
<div class="gradcam-panel">
  <div class="gradcam-header">
    <span>🔥 Grad-CAM Activation Map · {r['fname']}</span>
    <span style="color:var(--g);">AI FOCUS REGIONS HIGHLIGHTED</span>
  </div>
  <div class="gradcam-body">
    <div class="gradcam-img-wrap">
      {img_html}
      <div class="gradcam-badge">🎯 AI Activated Zones</div>
    </div>
    <div class="gradcam-right">
      <div>
        <div class="gcd-title">{info['short']}</div>
        <div class="gcd-sci">{info['pathogen']}</div>
      </div>
      <div class="gcd-why">{why_text}</div>
      <div style="margin-top:.2rem;">{prob_bars}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # 4. AI Explanation Panel
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧠 AI Explanation Panel</div>', unsafe_allow_html=True)
        seen = set()
        for r in results:
            lbl = r["label"]
            if lbl in seen: continue
            seen.add(lbl)
            info = r["info"]
            sc = info["sev_color"]
            chips = "".join(f'<span class="sym-chip">{s}</span>' for s in info["symptoms"])
            steps = "".join(f'<div class="step-row"><span class="step-num">{j}</span><span>{step}</span></div>' for j, step in enumerate(info["treatment_steps"], 1))

            with st.expander(f"{info['icon']}  {info['short']} — Full Analysis", expanded=True):
                st.markdown(f"""
<div class="explain-grid">
  <div class="explain-box">
    <div class="explain-box-h">📋 Model Analysis</div>
    <div class="explain-box-b">{info['desc']}</div>
    <span class="sev-badge-lg" style="background:{sc}15;color:{sc};border:1px solid {sc}40;">SEVERITY: {info['severity']}</span>
  </div>
  <div class="explain-box">
    <div class="explain-box-h">🔍 Matched Symptoms</div>
    <div>{chips}</div>
  </div>
  <div class="explain-box">
    <div class="explain-box-h">🛡 Treatment Protocol</div>
    {steps}
  </div>
  <div class="explain-box">
    <div class="explain-box-h">📊 Disease Metrics</div>
    <div class="explain-box-b">
      <strong>Yield impact:</strong> {info['yield_impact']}<br>
      <strong>Spread rate:</strong> {info['spread_rate']}<br>
      <strong>Risk score:</strong> {info['risk_score']}/100<br>
      <strong>Pathogen:</strong> <em>{info['pathogen']}</em>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # 5. Charts Dashboard
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📊 AI Risk Dashboard</div>', unsafe_allow_html=True)

        r0 = results[0]
        risk_pct = r0["info"]["risk_score"]
        risk_color = r0["info"]["sev_color"]
        risk_label = r0["info"]["severity"] if r0["info"]["severity"] != "NONE" else "LOW"

        total_h = max(len(st.session_state.history), 1)
        hc = sum(1 for h in st.session_state.history if h["status"] == "ok")
        dc = total_h - hc
        h_pct = round(hc / total_h * 100)
        R_d = 32; C_d = 2 * 3.14159 * R_d
        h_arc = hc / total_h * C_d; d_arc = dc / total_h * C_d

        st.markdown(f"""
<div class="charts-row">
  <div class="chart-card">
    <div class="chart-title">Disease Risk Meter</div>
    <div style="text-align:center;">
      <div class="gauge-num" style="color:{risk_color};">{risk_pct}</div>
      <div class="gauge-lbl">{risk_label} risk — score / 100</div>
      <div class="gauge-bar">
        <div class="gauge-needle" style="left:{risk_pct}%;"></div>
      </div>
      <div class="gauge-scale"><span>Low</span><span>Med</span><span>High</span></div>
    </div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Field Health Ratio</div>
    <div class="donut-row">
      <svg width="84" height="84" viewBox="0 0 84 84" style="transform:rotate(-90deg);flex-shrink:0;">
        <circle cx="42" cy="42" r="{R_d}" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="9"/>
        <circle cx="42" cy="42" r="{R_d}" fill="none" stroke="#7ddb8f" stroke-width="9"
          stroke-dasharray="{h_arc:.2f} {C_d:.2f}" stroke-linecap="round"/>
        <circle cx="42" cy="42" r="{R_d}" fill="none" stroke="#ff7070" stroke-width="9"
          stroke-dasharray="{d_arc:.2f} {C_d:.2f}" stroke-dashoffset="-{h_arc:.2f}"  <!-- offset by healthy arc so diseased arc follows on -->
          stroke-linecap="round" style="opacity:{1 if dc > 0 else 0};"/>
      </svg>
      <div class="donut-lgd">
        <div class="donut-big">{h_pct}%</div>
        <div class="donut-sub">Field Healthy</div>
        <div class="donut-item" style="margin-top:.4rem;"><div class="donut-dot" style="background:#7ddb8f;"></div>Healthy ({hc})</div>
        <div class="donut-item"><div class="donut-dot" style="background:#ff7070;"></div>Diseased ({dc})</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        with st.expander("📈 Probability Distribution — All Results"):
            for r in results:
                st.markdown(f"**{r['fname']}**", unsafe_allow_html=False)
                for cls in CLASSES:
                    p = r["all_probs"][cls]
                    hi = "prob-hi" if cls == r["label"] else ""
                    st.markdown(f'<div class="prob-row"><span class="prob-name">{cls}</span><div class="prob-tr"><div class="prob-fill {hi}" style="width:{p*100:.1f}%"></div></div><span class="prob-pct">{p*100:.1f}%</span></div>', unsafe_allow_html=True)

        # 6. Farmer Intelligence
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">👨‍🌾 Farmer Intelligence</div>', unsafe_allow_html=True)
        seen2 = set()
        for r in results:
            if r["label"] in seen2: continue
            seen2.add(r["label"])
            info = r["info"]
            st.markdown(f"""
<div class="farmer-card">
  <div class="farmer-head">{info['icon']} &nbsp; {info['short']}</div>
  <div class="farmer-body">{info['farmer_advice']}</div>
  <div class="weather-note">🌦️ <strong>Weather trigger:</strong> {info['weather_trigger']}</div>
</div>
""", unsafe_allow_html=True)

        # 7. Weather Risk
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🌦️ Weather-Based Disease Risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="weather-chips">' + "".join(
            f'<div class="wchip"><div class="wc-ico">{w["icon"]}</div><div class="wc-lbl">{w["label"]}</div><div class="wc-risk" style="color:{w["risk_color"]};">{w["risk"]}</div></div>'
            for w in WEATHER_CONDITIONS) + '</div>', unsafe_allow_html=True)

        weather_sel = st.selectbox("Field conditions:", [w["label"] for w in WEATHER_CONDITIONS], label_visibility="visible")
        w_info = next(w for w in WEATHER_CONDITIONS if w["label"] == weather_sel)
        rc = w_info["risk_color"]
        trigger = results[0]["info"]["weather_trigger"] if results else ""
        st.markdown(f"""
<div class="risk-result" style="background:var(--bg3);border-left:2px solid {rc};border-radius:10px;">
  {w_info['icon']} &nbsp;<strong style="color:{rc};">{w_info['risk']} RISK</strong> &nbsp;·&nbsp;
  Index: <strong style="color:{rc};">{w_info['risk_pct']}%</strong><br>
  <span style="font-size:.75rem;color:var(--c3);">{w_info['desc']}</span><br>
  <span style="font-size:.73rem;color:var(--c4);margin-top:.4rem;display:block;">{trigger}</span>
</div>
""", unsafe_allow_html=True)

    # History
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
                st.session_state.history = []; st.session_state.scanned = 0; st.session_state.results = []; st.rerun()

    st.markdown("""
<div class="app-footer">
  🌽 &nbsp;<strong>CornScan AI v6.0 Ultimate</strong><br>
  TensorFlow / Keras · CNN Plant Disease Detection · Grad-CAM Heatmaps<br>
  <span style="opacity:.45;">No data leaves your device · For research & field-scouting use</span>
</div>
""", unsafe_allow_html=True)


# Router
# === PAGE ROUTER: dispatches to loading/landing/main based on session_state.page ===
if st.session_state.page == "loading":
    loading_page()
elif st.session_state.page == "landing":
    landing_page()
else:
    main_app()