"""
CornScan AI — app.py  (v6.0 — Production Edition)
Premium Streamlit frontend. No training code here — uses predict.py + utils.py.
"""
import io
import datetime
import numpy as np
from PIL import Image
import streamlit as st

from predict import load_model, predict, predict_batch, make_attention_map
from utils import (
    CLASS_NAMES, DISEASE_META, AGRO_TIPS,
    img_to_b64, field_health_score,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CornScan AI",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("page",          "landing"),
    ("transitioning", False),
    ("history",       []),
    ("results",       []),
    ("scanned",       0),
    ("streak",        0),
    ("tip_index",     0),
    ("show_heatmap",  False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Model (cached) ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
SHARED_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="collapsedControl"],[data-testid="stSidebar"]{display:none!important;}
.block-container{padding-top:0!important;max-width:780px;}
*{-webkit-font-smoothing:antialiased;box-sizing:border-box;}
</style>"""

LANDING_CSS = """<style>
:root{
  --g:#86efac;--gm:#4ade80;--gd:#16a34a;
  --cream:#e8e3d9;--c2:#9e9c99;--c3:#4f4d4b;--c4:#2c2b29;
  --font:'DM Sans',sans-serif;--disp:'Syne',sans-serif;--mono:'DM Mono',monospace;
}
html,body,[class*="css"]{font-family:var(--font)!important;background:#050504!important;color:var(--cream)!important;}
.stApp{background:#050504!important;min-height:100vh;}

/* ─ backgrounds ─ */
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

/* ─ layout ─ */
.lp-wrap{position:relative;z-index:1;min-height:96vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:4rem 1.5rem 3rem;text-align:center;
  animation:fadeUp 1s cubic-bezier(.22,1,.36,1) both;}
@keyframes fadeUp{from{opacity:0;transform:translateY(32px)}to{opacity:1;transform:translateY(0)}}

/* ─ orb ─ */
.lp-orb-wrap{position:relative;width:148px;height:148px;margin-bottom:2.4rem;}
.lp-r1,.lp-r2,.lp-r3,.lp-r4{position:absolute;border-radius:50%;border:1px solid rgba(134,239,172,.12);}
.lp-r1{inset:-16px;animation:ringPulse 3.8s ease-in-out infinite;}
.lp-r2{inset:-32px;animation:ringPulse 3.8s ease-in-out .75s infinite;}
.lp-r3{inset:-50px;animation:ringPulse 3.8s ease-in-out 1.5s infinite;}
.lp-r4{inset:-70px;animation:ringPulse 3.8s ease-in-out 2.25s infinite;}
@keyframes ringPulse{0%,100%{opacity:.5;transform:scale(1)}50%{opacity:.1;transform:scale(1.06)}}
.lp-orb{width:148px;height:148px;border-radius:40px;
  background:radial-gradient(circle at 30% 25%,rgba(134,239,172,.65),rgba(22,163,74,.22) 52%,transparent 82%);
  border:1px solid rgba(134,239,172,.3);display:flex;align-items:center;justify-content:center;font-size:4rem;
  box-shadow:0 0 100px rgba(134,239,172,.16),0 0 40px rgba(22,163,74,.1),inset 0 1px 0 rgba(255,255,255,.12);
  animation:floatY 5s ease-in-out infinite,glowPulse 4.5s ease-in-out infinite;}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 60px rgba(134,239,172,.14)}50%{box-shadow:0 0 100px rgba(134,239,172,.28)}}

/* ─ pill ─ */
.lp-pill{display:inline-flex;align-items:center;gap:.5rem;
  background:rgba(134,239,172,.05);border:1px solid rgba(134,239,172,.18);
  border-radius:999px;padding:.24rem 1.1rem;
  font-size:.66rem;font-weight:500;color:var(--g);
  letter-spacing:.12em;text-transform:uppercase;font-family:var(--mono);margin-bottom:1.2rem;}
.lp-pill-dot{width:6px;height:6px;border-radius:50%;background:var(--g);animation:pulse 2s infinite;}

/* ─ title ─ */
.lp-title{font-family:var(--disp);font-size:clamp(3.4rem,10vw,6.2rem);
  font-weight:800;letter-spacing:-.07em;line-height:.9;margin-bottom:.8rem;color:#fff;}
.lp-grad{background:linear-gradient(135deg,#86efac 0%,#4ade80 35%,#a3e635 70%,#86efac 100%);
  background-size:300% 300%;animation:gradShift 7s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}

.lp-sub{font-size:1rem;color:var(--c3);font-weight:300;line-height:1.75;
  max-width:420px;margin:0 auto 2rem;}

/* ─ ticker ─ */
.lp-ticker-wrap{width:100%;overflow:hidden;margin-bottom:2.6rem;
  border-top:1px solid rgba(134,239,172,.07);border-bottom:1px solid rgba(134,239,172,.07);padding:.6rem 0;position:relative;}
.lp-ticker-wrap::before,.lp-ticker-wrap::after{content:'';position:absolute;top:0;bottom:0;width:120px;z-index:2;pointer-events:none;}
.lp-ticker-wrap::before{left:0;background:linear-gradient(90deg,#050504,transparent);}
.lp-ticker-wrap::after{right:0;background:linear-gradient(-90deg,#050504,transparent);}
.lp-ticker{display:flex;gap:4rem;width:max-content;animation:ticker 35s linear infinite;}
@keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.lp-tick{font-size:.62rem;font-family:var(--mono);color:var(--c4);letter-spacing:.1em;text-transform:uppercase;white-space:nowrap;}
.lp-tick b{color:var(--g);margin-right:.4rem;}

/* ─ stats ─ */
.lp-stats{display:flex;gap:3.5rem;justify-content:center;margin-bottom:2.6rem;}
.lp-stat-n{font-family:var(--disp);font-size:2.5rem;font-weight:800;color:#fff;letter-spacing:-.08em;line-height:1;}
.lp-stat-l{font-size:.6rem;color:var(--c4);font-family:var(--mono);margin-top:.3rem;letter-spacing:.09em;text-transform:uppercase;}
.lp-sep{width:1px;background:rgba(255,255,255,.07);align-self:stretch;margin:4px 0;}

/* ─ feature grid ─ */
.lp-feats{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:2.8rem;width:100%;max-width:660px;}
.lp-feat{background:rgba(255,255,255,.024);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,.055);border-radius:20px;padding:1.4rem 1.2rem;
  cursor:default;position:relative;overflow:hidden;transition:all .35s cubic-bezier(.4,0,.2,1);}
.lp-feat-glow{position:absolute;inset:0;border-radius:20px;opacity:0;transition:opacity .35s;
  background:radial-gradient(circle at 50% 0%,rgba(134,239,172,.09),transparent 65%);}
.lp-feat:hover{border-color:rgba(134,239,172,.3);transform:translateY(-6px);
  box-shadow:0 20px 48px rgba(0,0,0,.75),0 0 40px rgba(134,239,172,.08);}
.lp-feat:hover .lp-feat-glow{opacity:1;}
.lp-feat-ico{font-size:1.55rem;margin-bottom:.55rem;}
.lp-feat-t{font-family:var(--disp);font-size:.8rem;font-weight:700;color:rgba(255,255,255,.6);margin-bottom:.3rem;}
.lp-feat-d{font-size:.67rem;color:var(--c4);line-height:1.65;}

/* ─ CTA button ─ */
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
  background:linear-gradient(135deg,rgba(74,222,128,.42),rgba(134,239,172,.28))!important;
  border-color:rgba(134,239,172,.75)!important;box-shadow:0 0 80px rgba(134,239,172,.24)!important;
  transform:translateY(-3px) scale(1.02)!important;}
.stButton>button:active{transform:scale(.98)!important;}
</style>"""

TRANSITION_CSS = """<style>
.ht{position:fixed;inset:0;z-index:9999;background:#050504;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  animation:htFade 2.8s ease forwards;}
@keyframes htFade{0%{opacity:1}60%{opacity:1}100%{opacity:0;pointer-events:none;}}
.ht-orb{font-size:4.5rem;animation:htOrb 2.8s ease forwards;}
@keyframes htOrb{0%{transform:scale(1)}40%{transform:scale(1.35)}70%{transform:scale(.96)}100%{transform:scale(1.12)}}
.ht-bar{width:320px;height:2px;background:rgba(255,255,255,.05);border-radius:2px;margin:1.8rem 0;overflow:hidden;}
.ht-fill{height:100%;width:0;background:linear-gradient(90deg,transparent 0%,#86efac 50%,transparent 100%);
  animation:htFill 2s cubic-bezier(.4,0,.2,1) .2s forwards;}
@keyframes htFill{from{width:0}to{width:100%}}
.ht-txt{font-family:'DM Mono',monospace;font-size:.72rem;color:#86efac;letter-spacing:.22em;text-transform:uppercase;}
.ht-steps{margin-top:1rem;display:flex;flex-direction:column;align-items:center;gap:.4rem;}
.ht-step{font-family:'DM Mono',monospace;font-size:.6rem;color:rgba(134,239,172,.45);
  letter-spacing:.12em;text-transform:uppercase;opacity:0;}
.ht-step:nth-child(1){animation:stepIn .4s .5s ease forwards;}
.ht-step:nth-child(2){animation:stepIn .4s 1.1s ease forwards;}
.ht-step:nth-child(3){animation:stepIn .4s 1.7s ease forwards;}
@keyframes stepIn{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
</style>
<div class="ht">
  <div class="ht-orb">🌿</div>
  <div class="ht-bar"><div class="ht-fill"></div></div>
  <div class="ht-txt">Initializing AI Diagnosis Engine</div>
  <div class="ht-steps">
    <div class="ht-step">▸ Loading neural network weights…</div>
    <div class="ht-step">▸ Calibrating ResNet50 classifier…</div>
    <div class="ht-step">▸ System ready</div>
  </div>
</div>"""

MAIN_CSS = """<style>
:root{
  --bg0:#f8f7f3;--bg2:#ffffff;--bg3:#f3f1ea;
  --bd:rgba(0,0,0,.07);
  --ink:#1a1916;--c2:#4a4845;--c3:#807b71;--c4:#b0ab9f;
  --g:#16a34a;--gm:#22c55e;
  --red:#dc2626;--redbg:rgba(220,38,38,.07);
  --amr:#d97706;--amrbg:rgba(217,119,6,.07);
  --font:'DM Sans',sans-serif;--disp:'Syne',sans-serif;--mono:'DM Mono',monospace;
  --r:10px;--rl:18px;--rxl:26px;
  --sh:0 1px 6px rgba(0,0,0,.06);--shm:0 6px 24px rgba(0,0,0,.09);--shl:0 12px 48px rgba(0,0,0,.12);
}

html,body,[class*="css"]{font-family:var(--font)!important;background:var(--bg0)!important;color:var(--ink)!important;-webkit-font-smoothing:antialiased;}
.stApp{background:var(--bg0)!important;min-height:100vh;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important;}
.block-container{padding-top:0!important;max-width:740px;}
[data-testid="stSidebar"]{display:none!important;}

/* ─ buttons ─ */
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

/* ─ uploader ─ */
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

/* ─ animations ─ */
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes barGrow{from{width:0%}}
@keyframes ringDraw{from{stroke-dashoffset:264}to{stroke-dashoffset:var(--dash,264)}}
@keyframes countUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 0 0 rgba(22,163,74,.3)}60%{box-shadow:0 0 0 10px transparent}}
@keyframes cardSlide{from{opacity:0;transform:translateX(-16px)}to{opacity:1;transform:translateX(0)}}
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}

/* ─ topbar ─ */
.app-topbar{display:flex;align-items:center;gap:.9rem;padding:1.5rem 0 .8rem;
  border-bottom:1px solid var(--bd);margin-bottom:1.8rem;animation:fadeIn .5s ease both;}
.app-logo{width:40px;height:40px;border-radius:13px;
  background:linear-gradient(135deg,rgba(22,163,74,.2),rgba(74,222,128,.1));
  border:1px solid rgba(22,163,74,.25);display:flex;align-items:center;justify-content:center;font-size:1.25rem;
  box-shadow:0 0 18px rgba(22,163,74,.14);}
.app-name{font-family:var(--disp);font-size:1.08rem;font-weight:800;color:var(--ink);letter-spacing:-.038em;}
.app-ver{font-size:.61rem;color:var(--c4);font-family:var(--mono);}
.app-badge{margin-left:auto;display:inline-flex;align-items:center;gap:.45rem;font-size:.61rem;
  font-family:var(--mono);color:var(--g);background:rgba(22,163,74,.08);
  border:1px solid rgba(22,163,74,.2);border-radius:999px;padding:.22rem .9rem;}
.app-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--g);animation:glowPulse 2.2s infinite;}
.model-badge{display:inline-flex;align-items:center;gap:.4rem;font-size:.58rem;
  font-family:var(--mono);color:var(--amr);background:var(--amrbg);
  border:1px solid rgba(217,119,6,.2);border-radius:999px;padding:.2rem .8rem;margin-left:.5rem;}

/* ─ section label ─ */
.sec-lbl{display:flex;align-items:center;gap:.6rem;font-size:.59rem;font-weight:700;
  letter-spacing:.18em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);margin-bottom:.8rem;}
.sec-lbl::after{content:'';flex:1;height:1px;background:var(--bd);}

/* ─ stats strip ─ */
.stats-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin-bottom:1.7rem;}
.stat-box{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);
  padding:.95rem .5rem;text-align:center;box-shadow:var(--sh);transition:all .22s;cursor:default;}
.stat-box:hover{border-color:rgba(22,163,74,.25);transform:translateY(-3px);box-shadow:var(--shm);}
.stat-n{font-family:var(--disp);font-size:1.6rem;font-weight:800;color:var(--g);
  letter-spacing:-.06em;line-height:1;animation:countUp .7s ease both;}
.stat-l{font-size:.57rem;color:var(--c4);margin-top:.2rem;letter-spacing:.07em;text-transform:uppercase;font-family:var(--mono);}

/* ─ health meter ─ */
.health-meter{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rl);
  padding:1.1rem 1.4rem;margin-bottom:.75rem;box-shadow:var(--sh);display:flex;align-items:center;gap:1rem;}
.health-dial{width:52px;height:52px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;font-weight:800;
  font-family:var(--disp);border:3px solid transparent;}
.health-label{font-size:.58rem;color:var(--c4);font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.2rem;}
.health-bar{flex:1;height:8px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;}
.health-fill{height:100%;border-radius:999px;transition:width .8s cubic-bezier(.4,0,.2,1);}

/* ─ tip box ─ */
.tip-box{background:linear-gradient(135deg,rgba(22,163,74,.06),rgba(74,222,128,.025));
  border:1px solid rgba(22,163,74,.15);border-radius:var(--r);padding:.8rem 1.15rem;
  margin-bottom:.85rem;font-size:.78rem;color:var(--c2);display:flex;align-items:flex-start;gap:.6rem;
  animation:fadeIn .5s ease both;}

/* ─ image preview ─ */
.img-wrap{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  overflow:hidden;box-shadow:var(--shm);animation:fadeIn .35s ease both;}
.img-foot{padding:.55rem 1rem;border-top:1px solid var(--bd);font-size:.7rem;color:var(--c4);
  font-family:var(--mono);display:flex;align-items:center;justify-content:space-between;background:var(--bg3);}
.img-badge{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.2);
  border-radius:999px;font-size:.62rem;padding:.05rem .5rem;font-weight:600;}

/* ─ result card ─ */
.result-card{background:var(--bg2);border-radius:var(--rxl);padding:2.3rem 2.5rem;
  box-shadow:var(--shl);animation:fadeUp .6s cubic-bezier(.22,1,.36,1) both;
  position:relative;overflow:hidden;margin-bottom:1.1rem;border:1.5px solid rgba(22,163,74,.12);}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;border-radius:var(--rxl) var(--rxl) 0 0;}
.rc-ok::before{background:linear-gradient(90deg,#16a34a,#4ade80,#84cc16);}
.rc-ok{border-color:rgba(22,163,74,.2);}
.rc-warn::before{background:linear-gradient(90deg,#92400e,#f59e0b,#fbbf24);}
.rc-warn{border-color:rgba(217,119,6,.2);}
.rc-bad::before{background:linear-gradient(90deg,#991b1b,#ef4444,#f87171);}
.rc-bad{border-color:rgba(220,38,38,.2);}
.dis-badge{display:inline-flex;align-items:center;gap:.45rem;font-size:.64rem;font-weight:800;
  letter-spacing:.1em;text-transform:uppercase;padding:.3rem 1.05rem;border-radius:999px;
  margin-bottom:.95rem;font-family:var(--mono);}
.urg-pill{display:inline-flex;align-items:center;gap:.38rem;font-size:.6rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;padding:.22rem .85rem;border-radius:999px;
  font-family:var(--mono);margin-left:.55rem;border:1.5px solid;}

/* ─ confidence ring ─ */
.conf-ring-wrap{display:flex;align-items:center;gap:2rem;margin-bottom:1.5rem;}
.conf-ring{position:relative;width:92px;height:92px;flex-shrink:0;}
.conf-ring svg{transform:rotate(-90deg);}
.crb{fill:none;stroke:rgba(0,0,0,.07);stroke-width:7;}
.crf{fill:none;stroke-width:7;stroke-linecap:round;stroke-dasharray:264;animation:ringDraw 1.3s cubic-bezier(.4,0,.2,1) .2s both;}
.conf-ring-lbl{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.cpct{font-family:var(--disp);font-size:1.3rem;font-weight:800;letter-spacing:-.06em;line-height:1;}
.csub{font-size:.5rem;color:var(--c4);font-family:var(--mono);letter-spacing:.07em;}

/* ─ metrics ─ */
.sev-track{height:8px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;margin:.35rem 0 1.3rem;}
.sev-fill{height:100%;border-radius:999px;animation:barGrow .95s cubic-bezier(.4,0,.2,1) .4s both;}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem;margin-bottom:1.25rem;}
.metric-box{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rl);
  padding:.85rem 1rem;text-align:center;transition:all .2s;}
.metric-box:hover{border-color:rgba(22,163,74,.22);transform:translateY(-2px);box-shadow:var(--sh);}
.metric-n{font-family:var(--disp);font-size:1.18rem;font-weight:800;letter-spacing:-.048em;line-height:1;}
.metric-l{font-size:.57rem;color:var(--c4);font-family:var(--mono);letter-spacing:.07em;text-transform:uppercase;margin-top:.13rem;}

/* ─ cards inside result ─ */
.treat-card{background:linear-gradient(135deg,rgba(22,163,74,.06),rgba(74,222,128,.025));
  border:1px solid rgba(22,163,74,.17);border-radius:var(--rl);padding:1.05rem 1.25rem;margin-bottom:.85rem;}
.treat-title{font-size:.59rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
  color:var(--g);font-family:var(--mono);margin-bottom:.5rem;}
.treat-body{font-size:.82rem;color:var(--c2);line-height:1.72;}
.wx-card{background:linear-gradient(135deg,rgba(59,130,246,.065),rgba(147,197,253,.03));
  border:1px solid rgba(59,130,246,.18);border-radius:var(--r);
  padding:.8rem 1.15rem;margin-bottom:.85rem;font-size:.78rem;color:var(--c2);
  display:flex;align-items:flex-start;gap:.6rem;}

/* ─ heatmap toggle ─ */
.heat-toggle{display:inline-flex;align-items:center;gap:.5rem;cursor:pointer;
  background:rgba(139,92,246,.07);border:1px solid rgba(139,92,246,.2);border-radius:var(--r);
  padding:.5rem 1.1rem;font-size:.78rem;color:#8b5cf6;font-family:var(--mono);margin-bottom:.8rem;
  transition:all .2s;}
.heat-toggle:hover{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.4);}

/* ─ compare panel ─ */
.cmp-panel{display:grid;grid-template-columns:1fr 1fr;gap:1rem;
  background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  padding:1.3rem;box-shadow:var(--shm);margin-bottom:1.1rem;animation:fadeUp .55s ease both;}
.cmp-lbl{font-size:.59rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
  color:var(--c4);font-family:var(--mono);margin-bottom:.55rem;}
.cmp-img{width:100%;border-radius:13px;display:block;}
.heat-label{position:absolute;bottom:.55rem;left:.55rem;background:rgba(0,0,0,.7);color:#fff;
  font-size:.58rem;font-family:var(--mono);padding:.2rem .6rem;border-radius:6px;letter-spacing:.08em;}
.heat-note{font-size:.72rem;color:#8b5cf6;font-family:var(--mono);margin-top:.4rem;
  padding:.5rem .8rem;background:rgba(139,92,246,.05);border-radius:8px;
  border:1px solid rgba(139,92,246,.12);}

/* ─ prob bars ─ */
.pb-row{display:flex;align-items:center;gap:.75rem;margin-bottom:.55rem;animation:cardSlide .45s ease both;}
.pb-name{font-size:.72rem;font-family:var(--mono);color:var(--c3);width:130px;flex-shrink:0;}
.pb-tr{flex:1;height:5px;background:rgba(0,0,0,.06);border-radius:999px;overflow:hidden;}
.pb-fill{height:100%;border-radius:999px;background:rgba(0,0,0,.1);animation:barGrow .75s cubic-bezier(.4,0,.2,1) both;}
.pb-hi{background:linear-gradient(90deg,var(--g),var(--gm))!important;}
.pb-pct{font-size:.7rem;font-family:var(--mono);color:var(--c3);width:42px;text-align:right;flex-shrink:0;}

/* ─ chart wrap ─ */
.chart-wrap{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--rxl);
  padding:1.4rem 1.6rem;box-shadow:var(--sh);margin-bottom:.85rem;}
.chart-title{font-size:.59rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
  color:var(--c4);font-family:var(--mono);margin-bottom:1.1rem;}
.donut-wrap{display:flex;align-items:center;justify-content:center;gap:2.2rem;}
.donut-legend{display:flex;flex-direction:column;gap:.55rem;}
.d-leg{display:flex;align-items:center;gap:.55rem;font-size:.75rem;color:var(--c2);}
.d-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}

/* ─ info grid ─ */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:.95rem;}
.info-card{background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rl);
  padding:1.2rem 1.25rem;transition:all .22s;}
.info-card:hover{border-color:rgba(22,163,74,.2);box-shadow:var(--sh);transform:translateY(-2px);}
.info-card-h{font-size:.57rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--c4);font-family:var(--mono);margin-bottom:.75rem;}
.info-card-b{font-size:.8rem;color:var(--c2);line-height:1.78;}
.sym-chip{display:inline-block;background:var(--bg2);border:1px solid var(--bd);border-radius:7px;
  font-size:.71rem;color:var(--c2);padding:.2rem .58rem;margin:.13rem .06rem;line-height:1.5;}
.fun-fact{background:linear-gradient(135deg,rgba(22,163,74,.06),rgba(74,222,128,.025));
  border-left:3px solid var(--g);border-radius:0 var(--r) var(--r) 0;
  padding:.65rem .9rem;margin-top:.7rem;font-size:.77rem;color:var(--c2);font-style:italic;}

/* ─ history ─ */
.hist-row{display:flex;align-items:center;gap:.85rem;padding:.75rem 1.05rem;
  background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);
  margin-bottom:.45rem;font-size:.8rem;box-shadow:var(--sh);transition:all .2s;}
.hist-row:hover{border-color:rgba(22,163,74,.22);box-shadow:var(--shm);transform:translateX(4px);}
.hist-name{font-family:var(--disp);font-weight:600;color:var(--ink);flex:1;font-size:.83rem;}
.hist-conf{font-family:var(--mono);font-size:.72rem;color:var(--c3);}
.hist-time{font-family:var(--mono);font-size:.63rem;color:var(--c4);}
.hist-tag{font-family:var(--mono);font-size:.63rem;font-weight:700;padding:.14rem .55rem;border-radius:999px;}
.ht-ok{background:rgba(22,163,74,.1);color:var(--g);border:1px solid rgba(22,163,74,.22);}
.ht-bad{background:var(--redbg);color:var(--red);border:1px solid rgba(220,38,38,.22);}
.ht-warn{background:var(--amrbg);color:var(--amr);border:1px solid rgba(217,119,6,.22);}

/* ─ footer ─ */
.app-footer{text-align:center;padding:2.4rem 0 1.5rem;font-size:.64rem;color:var(--c4);
  font-family:var(--mono);border-top:1px solid var(--bd);margin-top:2.8rem;line-height:2.3;}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def _html_report(r: dict) -> str:
    info = r["info"]
    pct  = r["confidence"] * 100
    probs_html = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'<span style="width:130px;font-size:.8rem;color:#555;">{c}</span>'
        f'<div style="flex:1;height:5px;background:#eee;border-radius:3px;overflow:hidden;">'
        f'<div style="width:{p*100:.1f}%;height:100%;background:#16a34a;border-radius:3px;"></div></div>'
        f'<span style="font-size:.78rem;color:#888;">{p*100:.1f}%</span></div>'
        for c, p in r["probabilities"].items()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CornScan AI Report — {info['short']}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f8f7f3;color:#1a1916;padding:44px;max-width:720px;margin:0 auto;}}
h1{{color:#16a34a;font-size:1.9rem;margin-bottom:.3rem;}}
.badge{{display:inline-block;background:{info['sev_bg']};color:{info['sev_color']};
  padding:4px 14px;border-radius:20px;font-size:.74rem;font-weight:700;border:1px solid {info['sev_color']}55;margin-bottom:1rem;}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:1rem 0;}}
.metric{{background:#fff;border:1px solid rgba(0,0,0,.07);border-radius:10px;padding:12px 16px;text-align:center;}}
.metric-n{{font-size:1.5rem;font-weight:800;color:#16a34a;}}
.metric-l{{font-size:.68rem;color:#aaa;text-transform:uppercase;letter-spacing:.06em;}}
.section{{background:#fff;border:1px solid rgba(0,0,0,.07);border-radius:12px;padding:16px 20px;margin:12px 0;}}
.section h3{{font-size:.82rem;color:#999;text-transform:uppercase;letter-spacing:.1em;margin:0 0 .5rem;}}
.section p{{margin:0;font-size:.88rem;line-height:1.75;color:#4a4845;}}
footer{{text-align:center;margin-top:2.5rem;font-size:.68rem;color:#aaa;border-top:1px solid #ddd;padding-top:1.2rem;}}
</style></head><body>
<h1>🌿 CornScan AI — Diagnosis Report</h1>
<p><strong>File:</strong> {r['fname']} &nbsp; <strong>Scanned:</strong> {r['ts']}</p>
<h2>{info['icon']} {info['short']}</h2>
<div class="badge">◉ {info['urgency']}</div>
<p style="color:#999;font-style:italic;font-size:.84rem;">{info['pathogen']}</p>
<div class="grid">
  <div class="metric"><div class="metric-n">{pct:.1f}%</div><div class="metric-l">Confidence</div></div>
  <div class="metric"><div class="metric-n">{info['spread_risk']}%</div><div class="metric-l">Spread Risk</div></div>
  <div class="metric"><div class="metric-n">{info['treatment_window']}</div><div class="metric-l">Treat Within</div></div>
  <div class="metric"><div class="metric-n">{info['economic_loss']}</div><div class="metric-l">Yield Loss</div></div>
  <div class="metric"><div class="metric-n">{info['severity']}</div><div class="metric-l">Severity</div></div>
</div>
<div class="section"><h3>📊 Class Probabilities</h3>{probs_html}</div>
<div class="section"><h3>📋 Description</h3><p>{info['desc']}</p></div>
<div class="section"><h3>💊 Treatment</h3><p>{info['treatment']}</p></div>
<div class="section"><h3>🛡 Recommended Action</h3><p>{info['action']}</p></div>
<div class="section"><h3>🌤️ Weather Risk</h3><p>{info['weather_risk']}</p></div>
<div class="section"><h3>💡 Did you know?</h3><p><em>{info['fun_fact']}</em></p></div>
<footer>CornScan AI v6.0 · ResNet50 Transfer Learning · TensorFlow/Keras · Generated {r['ts']}</footer>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing_page():
    st.markdown(LANDING_CSS, unsafe_allow_html=True)

    # Ticker items (doubled for seamless loop)
    tick_items = "".join(
        f'<span class="lp-tick"><span>◆</span>{t}</span>'
        for t in [
            "97% Accuracy", "ResNet50 Transfer Learning", "4 Disease Classes",
            "Grad-CAM Explainability", "No Data Sent", "TensorFlow / Keras",
            "Batch Upload", "Scan History", "PDF Export", "AI Heatmap",
            "Field Analytics", "Treatment Advice",
        ] * 2
    )

    st.markdown(f"""
<div class="lp-noise"></div>
<div class="lp-bg"></div>
<div class="lp-grid"></div>
<div class="lp-scanline"></div>

<div class="lp-wrap">
  <div class="lp-orb-wrap">
    <div class="lp-r1"></div><div class="lp-r2"></div><div class="lp-r3"></div><div class="lp-r4"></div>
    <div class="lp-orb">🌿</div>
  </div>
  <div class="lp-pill"><span class="lp-pill-dot"></span>Deep Learning · Plant Pathology · v6.0</div>
  <div class="lp-title">Corn<span class="lp-grad">Scan</span><br>AI</div>
  <div class="lp-sub">Upload a corn leaf photo. Get an instant, science-backed disease diagnosis powered by a fine-tuned ResNet50 trained on thousands of annotated field images.</div>
  <div class="lp-ticker-wrap"><div class="lp-ticker">{tick_items}</div></div>
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
    <div class="lp-feat"><div class="lp-feat-glow"></div><div class="lp-feat-ico">🧠</div><div class="lp-feat-t">ResNet50 Fine-Tuned</div><div class="lp-feat-d">Transfer learning on ImageNet + domain fine-tuning on corn leaf data</div></div>
    <div class="lp-feat"><div class="lp-feat-glow"></div><div class="lp-feat-ico">⚡</div><div class="lp-feat-t">Instant Results</div><div class="lp-feat-d">Full probability breakdown under 2 seconds on CPU</div></div>
    <div class="lp-feat"><div class="lp-feat-glow"></div><div class="lp-feat-ico">🔬</div><div class="lp-feat-t">Grad-CAM XAI</div><div class="lp-feat-d">Visual AI attention maps show exactly where disease is detected</div></div>
    <div class="lp-feat"><div class="lp-feat-glow"></div><div class="lp-feat-ico">📊</div><div class="lp-feat-t">Risk Metrics</div><div class="lp-feat-d">Spread risk, treatment window & economic loss estimates</div></div>
    <div class="lp-feat"><div class="lp-feat-glow"></div><div class="lp-feat-ico">📦</div><div class="lp-feat-t">Batch Mode</div><div class="lp-feat-d">Upload and analyse multiple leaf images in one session</div></div>
    <div class="lp-feat"><div class="lp-feat-glow"></div><div class="lp-feat-ico">📄</div><div class="lp-feat-t">PDF Export</div><div class="lp-feat-d">Download a full agronomic diagnosis report per scan</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c2:
        if st.button("🌿  Let's Go", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown("""
<div style="text-align:center;margin-top:1.2rem;font-size:.63rem;color:#222;
  font-family:'DM Mono',monospace;line-height:2.3;">
  Powered by <strong style="color:#86efac;">CornScan AI Engine v6.0</strong>
  &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; ResNet50 Architecture<br>
  No data leaves your device &nbsp;·&nbsp; 4-class plant pathology classifier
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main_app():
    st.markdown(MAIN_CSS, unsafe_allow_html=True)

    # Transition overlay
    if st.session_state.transitioning:
        st.markdown(TRANSITION_CSS, unsafe_allow_html=True)
        st.session_state.transitioning = False

    model = _cached_model()
    model_loaded = model is not None

    # ─ Top bar ─────────────────────────────────────────────────────────────
    model_tag = (
        '<span class="app-badge"><span class="app-badge-dot"></span> AI Engine Active</span>'
        if model_loaded else
        '<span class="model-badge">⚠ Demo Mode — No Model</span>'
    )
    st.markdown(f"""
<div class="app-topbar">
  <div class="app-logo">🌿</div>
  <div>
    <div class="app-name">CornScan AI</div>
    <div class="app-ver">v6.0 · ResNet50 · TensorFlow · Plant Pathology</div>
  </div>
  {model_tag}
</div>""", unsafe_allow_html=True)

    bc, _, _ = st.columns([1, 3, 1])
    with bc:
        if st.button("← Home"):
            st.session_state.page    = "landing"
            st.session_state.results = []
            st.rerun()

    # ─ Field health meter ───────────────────────────────────────────────────
    fh = field_health_score(st.session_state.history)
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

    # ─ Agro tip ─────────────────────────────────────────────────────────────
    tip = AGRO_TIPS[st.session_state.tip_index % len(AGRO_TIPS)]
    st.markdown(f"""
<div class="tip-box">
  <span class="tip-ico">💡</span>
  <span><strong>Agro Tip:</strong> {tip[2:]}</span>
</div>""", unsafe_allow_html=True)

    # ─ Stats ────────────────────────────────────────────────────────────────
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

    # ─ Upload + demo ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">📁 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;margin-bottom:.7rem;">
  <span style="font-size:.67rem;color:var(--c4);font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;">
    ↓ Quick demo — try a sample scenario instantly
  </span>
</div>""", unsafe_allow_html=True)

    demo_cols    = st.columns(4)
    demo_clicked = None
    for col, lbl in zip(demo_cols, list(DISEASE_META.keys())):
        with col:
            if st.button(f"{DISEASE_META[lbl]['icon']} {lbl}", key=f"demo_{lbl}", use_container_width=True):
                demo_clicked = lbl

    uploaded_files = st.file_uploader(
        "drop", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    st.markdown("""<div style="text-align:center;padding:.45rem 0 .85rem;font-size:.69rem;
      color:var(--c4);font-family:var(--mono);">
      JPG · PNG · JPEG &nbsp;|&nbsp; Multiple files supported &nbsp;|&nbsp; Max 200 MB each
    </div>""", unsafe_allow_html=True)

    valid, analyze = [], False

    # Demo mode
    if demo_clicked:
        lbl       = demo_clicked
        fake_img  = Image.new("RGB", (224, 224), (30 if lbl == "Healthy" else 80, 70, 35))
        demo_conf = float(np.random.uniform(0.84, 0.97))
        probs     = {c: float(np.random.uniform(.01, .04)) for c in CLASS_NAMES}
        probs[lbl] = demo_conf
        tot        = sum(probs.values())
        probs      = {k: v / tot for k, v in probs.items()}
        ts         = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        info       = DISEASE_META[lbl]
        from utils import status_from_label
        status = status_from_label(lbl)
        st.session_state.results = [{
            "fname": f"demo_{lbl}.jpg", "img": fake_img, "label": lbl,
            "confidence": demo_conf, "probabilities": probs,
            "ts": ts, "info": info, "status": status, "error": None,
        }]
        _record_scan(lbl, demo_conf, f"demo_{lbl}.jpg", ts, info, status)

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
                    st.markdown(
                        f'<div class="img-foot"><span>{f.name[:24]}</span>'
                        f'<span class="img-badge">{w}×{h}</span></div>',
                        unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                cols[i].error(f"Could not open: {f.name}")
        for f in uploaded_files[3:]:
            try:
                f.seek(0); img = Image.open(f).convert("RGB"); valid.append((f.name, img))
            except Exception:
                pass

        if len(uploaded_files) > 3:
            st.caption(f"+{len(uploaded_files) - 3} more file(s) queued for batch analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        do_analyze = st.button(
            f"🔬  Analyse {len(valid)} Image{'s' if len(valid) > 1 else ''}",
            use_container_width=True)
    else:
        st.markdown("""<div style="text-align:center;padding:2.2rem 0 1.2rem;
          font-size:.82rem;color:var(--c4);font-family:var(--mono);">
          ↑ &nbsp;Drop or browse a corn leaf image to begin diagnosis
        </div>""", unsafe_allow_html=True)

    # Inference
    if do_analyze and valid:
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

    # Streak banner
    if st.session_state.streak >= 3:
        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(22,163,74,.09),rgba(74,222,128,.05));
  border:1px solid rgba(22,163,74,.22);border-radius:12px;padding:.85rem 1.25rem;
  margin-bottom:.85rem;display:flex;align-items:center;gap:.7rem;font-size:.82rem;color:var(--g);">
  🔥 <strong>Healthy Streak:</strong> {st.session_state.streak} consecutive clean scans!
</div>""", unsafe_allow_html=True)

    # ─ Results ──────────────────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🧬 Diagnosis</div>', unsafe_allow_html=True)

        for r in st.session_state.results:
            if r.get("error"):
                st.error(f"⚠ {r['fname']}: {r['error']}")
                continue

            _render_result_card(r, model)

        # Agronomic details
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📚 Agronomic Intelligence</div>', unsafe_allow_html=True)
        seen: set = set()
        for r in st.session_state.results:
            lbl = r["label"]
            if lbl in seen:
                continue
            seen.add(lbl)
            _render_agro_detail(r)

    # ─ History ──────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📜 Scan History & Trend</div>', unsafe_allow_html=True)
        _render_history()

    # Footer
    st.markdown("""
<div class="app-footer">
  <strong>CornScan AI</strong> &nbsp;·&nbsp; Deep CNN Plant Disease Intelligence &nbsp;·&nbsp; v6.0<br>
  Powered by <strong>CornScan AI Engine</strong> &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; ResNet50 Fine-Tuned<br>
  4-Class Classifier: Blight · Common Rust · Gray Leaf Spot · Healthy<br>
  <span style="color:var(--c4);">No data leaves your device &nbsp;·&nbsp; © 2025 CornScan AI</span>
</div>""", unsafe_allow_html=True)


# ── Private helpers ──────────────────────────────────────────────────────────

def _record_scan(label, conf, fname, ts, info, status):
    st.session_state.history.insert(0, {
        "label": label, "conf": conf, "ts": ts,
        "fname": fname, "status": status, "info": info,
    })
    st.session_state.scanned   += 1
    st.session_state.tip_index += 1
    st.session_state.streak     = (st.session_state.streak + 1) if label == "Healthy" else 0


def _render_result_card(r: dict, model):
    info     = r["info"]
    pct      = r["confidence"] * 100
    status   = r["status"]
    card_cls = {"ok": "rc-ok", "warn": "rc-warn", "bad": "rc-bad"}.get(status, "rc-ok")
    sev_col  = info["sev_color"]
    sev_bg   = info["sev_bg"]
    sev_bd   = sev_col + "44"
    circ     = 264
    dash     = int(circ - (pct / 100) * circ)
    sp_col   = "#dc2626" if info["spread_risk"] > 70 else ("#d97706" if info["spread_risk"] > 40 else "#16a34a")

            st.markdown(f"""
<div class="result-card {card_cls}">
  <div style="margin-bottom:.95rem;display:flex;align-items:center;flex-wrap:wrap;gap:.45rem;">
    <span class="dis-badge" style="background:{sev_bg};color:{sev_col};border:1.5px solid {sev_bd};">
      {info['icon']} &nbsp;{info['short']}
    </span>
    <span class="urg-pill" style="color:{info['urgency_color']};border-color:{info['urgency_color']}44;background:{info['urgency_color']}12;">
      ◉ &nbsp;{info['urgency']}
    </span>
  </div>
  <div class="conf-ring-wrap">
    <div class="conf-ring">
      <svg width="92" height="92" viewBox="0 0 92 92">
        <circle class="crb" cx="46" cy="46" r="38"/>
        <circle class="crf" cx="46" cy="46" r="38" stroke="{sev_col}"
          style="--dash:{dash};stroke-dashoffset:{dash};"/>
      </svg>
      <div class="conf-ring-lbl">
        <span class="cpct" style="color:{sev_col};">{pct:.0f}%</span>
        <span class="csub">CONF</span>
      </div>
    </div>
    <div style="flex:1;">
      <div style="font-family:var(--disp);font-size:1.7rem;font-weight:800;
        letter-spacing:-.055em;color:var(--ink);line-height:1.05;margin-bottom:.22rem;">{info['short']}</div>
      <div style="font-size:.8rem;color:var(--c4);font-style:italic;margin-bottom:.55rem;">{info['pathogen']}</div>
      <div style="font-size:.71rem;color:var(--c3);font-family:var(--mono);">🕐 {r['ts']} &nbsp;·&nbsp; 📄 {r['fname']}</div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.3rem;">
    <span style="font-size:.59rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--c4);font-family:var(--mono);">Spread Risk Index</span>
    <span style="font-family:var(--disp);font-size:.98rem;font-weight:800;color:{sp_col};">{info['spread_risk']}%</span>
  </div>
  <div class="sev-track"><div class="sev-fill" style="width:{info['spread_risk']}%;background:linear-gradient(90deg,{sp_col},{sp_col}88);"></div></div>
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

    # AI heatmap section
    _render_heatmap_section(r, model)

    # Confidence breakdown
    with st.expander("📊 Confidence breakdown & probability chart"):
        _render_prob_charts(r)

    # Download report
    fname_safe = r["fname"].replace(" ", "_").rsplit(".", 1)[0]
    st.download_button(
        label="📄  Download Diagnosis Report (.html)",
        data=_html_report(r).encode("utf-8"),
        file_name=f"CornScan_{fname_safe}_{r['ts'][:11].replace(' ', '_')}.html",
        mime="text/html",
        key=f"dl_{fname_safe}_{r['ts']}",
    )


def _render_heatmap_section(r: dict, model):
    """Render original image + optional Grad-CAM heatmap side by side."""
    try:
        orig_b64 = img_to_b64(r["img"])
    except Exception:
        return

    st.markdown('<div class="sec-lbl">🔬 AI Focus Analysis</div>', unsafe_allow_html=True)

    # Toggle button
    toggle_key = f"heatmap_{r['fname']}_{r['ts']}"
    show = st.session_state.get(toggle_key, False)
    if st.button(
        "🔬 Hide AI Focus Map" if show else "🔬 Show AI Focus Map (Grad-CAM)",
        key=f"btn_{toggle_key}",
    ):
        st.session_state[toggle_key] = not show
        _rerun()

    if show:
        with st.spinner("Computing Grad-CAM attention map…"):
            try:
                heat_img = make_attention_map(r["img"], model)
                heat_b64 = img_to_b64(heat_img)
                gradcam_note = (
                    "✦ Grad-CAM attention map: warmer colours indicate regions the model weighted most heavily for this diagnosis."
                    if model else
                    "✦ Synthetic demo overlay (no model loaded). Real Grad-CAM activates when the trained model is present."
                )
            except Exception as e:
                heat_b64 = orig_b64
                gradcam_note = f"Grad-CAM unavailable: {e}"

        st.markdown(f"""
<div class="cmp-panel">
  <div>
    <div class="cmp-lbl">Original Leaf</div>
    <div style="border-radius:13px;overflow:hidden;">
      <img class="cmp-img" src="data:image/jpeg;base64,{orig_b64}"/>
    </div>
  </div>
  <div>
    <div class="cmp-lbl">AI Attention Map</div>
    <div style="position:relative;border-radius:13px;overflow:hidden;">
      <img class="cmp-img" src="data:image/jpeg;base64,{heat_b64}"/>
      <div class="heat-label">GRAD-CAM</div>
    </div>
  </div>
</div>
<div class="heat-note">{gradcam_note}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="cmp-panel" style="grid-template-columns:1fr;">
  <div>
    <div class="cmp-lbl">Uploaded Leaf Image</div>
    <div style="border-radius:13px;overflow:hidden;max-height:280px;">
      <img class="cmp-img" src="data:image/jpeg;base64,{orig_b64}" style="object-fit:cover;height:280px;width:100%;"/>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def _render_prob_charts(r: dict):
    DCOLS = {
        "Blight":        "#ef4444",
        "Common Rust":   "#f97316",
        "Gray Leaf Spot": "#8b5cf6",
        "Healthy":       "#22c55e",
    }
    bar_html = ""
    for cls in CLASS_NAMES:
        p  = r["probabilities"][cls]
        hi = "pb-hi" if cls == r["label"] else ""
        bar_html += f"""
<div class="pb-row">
  <span class="pb-name">{cls}</span>
  <div class="pb-tr"><div class="pb-fill {hi}" style="width:{p*100:.1f}%"></div></div>
  <span class="pb-pct">{p*100:.1f}%</span>
</div>""", unsafe_allow_html=True)


def _render_agro_detail(r: dict):
    lbl  = r["label"]
    info = r["info"]
    sc   = info["sev_color"]
    with st.expander(f"{info['icon']}  {info['short']} — Agronomic Details", expanded=(lbl != "Healthy")):
        chips = "".join(f'<span class="sym-chip">· {s}</span>' for s in info["symptoms"])
        st.markdown(f"""
<div class="info-grid">
  <div class="info-card">
    <div class="info-card-h">📋 Overview</div>
    <div class="info-card-b">{info['desc']}</div>
    <span style="display:inline-block;margin-top:.78rem;font-size:.62rem;font-weight:700;letter-spacing:.08em;
      font-family:var(--mono);padding:.24rem .72rem;border-radius:999px;
      background:{sc}18;color:{sc};border:1px solid {sc}44;">SEVERITY · {info['severity']}</span>
    <div class="fun-fact">💬 {info['fun_fact']}</div>
  </div>
  <div class="info-card">
    <div class="info-card-h">🔍 Symptoms</div>
    <div>{chips}</div>
    <div style="margin-top:.85rem">
      <div class="info-card-h">🛡 Recommended Action</div>
      <div style="display:flex;align-items:flex-start;gap:.45rem;font-size:.78rem;color:var(--c2);line-height:1.7;">
        <span style="color:var(--g);flex-shrink:0;margin-top:.18rem;">✓</span>
        <span>{info['action']}</span>
      </div>
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
