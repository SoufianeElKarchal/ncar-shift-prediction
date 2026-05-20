"""
=============================================================
  INTERFACE STREAMLIT — PRÉDICTION SHIFTS NCAR
  PFE : Optimisation du pilotage et de l'équilibrage NCAR

  Lancement : streamlit run ncar_app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, sys
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(__file__))
from ncar_model import (
    entrainer_modele, predire_shifts, repartir_shifts,
    calculer_shifts_optimises, cap_shift_reel, cap_demie_equipe,
    DONNEES_REELLES, make_features, FEATURES,
    STOCK_SEC, CAP_H_REF63, CAP_H_REF64, SHIFT_HEURES,
    OPE_COMPLET, OPE_DEMI, JOURS_SEMAINE
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NCAR — Shifts", page_icon="🏭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #EFF6FA;
}
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #D9E8F0;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #4A9BBF 0%, #2B7FA8 50%, #1B6690 100%);
    color: white;
    padding: 32px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(43, 127, 168, 0.25);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 240px; height: 240px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.main-header h1 {
    margin: 0;
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.main-header p {
    margin: 8px 0 0;
    font-size: 13px;
    opacity: 0.78;
    font-weight: 300;
    letter-spacing: 0.03em;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid rgba(74, 155, 191, 0.15);
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(43, 127, 168, 0.08);
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 8px 30px rgba(43, 127, 168, 0.14);
}
.kpi-label {
    font-size: 11px;
    color: #7FA8BE;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.kpi-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 36px;
    font-weight: 700;
    margin: 6px 0 0;
    line-height: 1;
}
.kpi-sub {
    font-size: 11px;
    margin: 6px 0 0;
    color: #9BBDCC;
    font-weight: 300;
}

/* ── Shift Cards ── */
.shift-matin {
    background: #EBF5FB;
    border-left: 3px solid #2B7FA8;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
}
.shift-apm {
    background: #EAF6F0;
    border-left: 3px solid #2FAE7A;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
}
.shift-demi {
    background: #FDF8EE;
    border-left: 3px solid #E4A82A;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
}
.shift-repos {
    background: #F4F7F9;
    border-left: 3px solid #C5D5DE;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    opacity: 0.55;
    font-family: 'DM Sans', sans-serif;
}

/* ── Section Titles ── */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 18px;
    font-weight: 600;
    color: #1B6690;
    border-left: 3px solid #4A9BBF;
    padding-left: 12px;
    margin: 24px 0 14px;
    letter-spacing: 0.01em;
}

/* ── Badges ── */
.badge-low {
    background: #FDECEA;
    color: #B03020;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-ok {
    background: #E4F5EC;
    color: #1A6640;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Goulot Cards ── */
.goulot-card {
    background: #FFF8EE;
    border: 1px solid #F0C472;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Tabs styling ── */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #7FA8BE !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: #1B6690 !important;
    border-bottom-color: #4A9BBF !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #1B6690; font-family: \"Cormorant Garamond\", serif; margin-top: 50px;'>🔒 NCAR Portal</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Identifiant")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "password123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")
    st.stop()

# ─────────────────────────────────────────────
# FICHIER SUIVI JOURNALIER & YAMAZUMI (persistance JSON)
# ─────────────────────────────────────────────
SUIVI_FILE = os.path.join(os.path.dirname(__file__), 'ncar_suivi.json')
YAMAZUMI_FILE = os.path.join(os.path.dirname(__file__), 'ncar_yamazumi.json')

def charger_suivi():
    if os.path.exists(SUIVI_FILE):
        with open(SUIVI_FILE) as f:
            return json.load(f)
    return {}

def sauvegarder_suivi(data):
    with open(SUIVI_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def charger_yamazumi():
    if os.path.exists(YAMAZUMI_FILE):
        with open(YAMAZUMI_FILE) as f:
            return json.load(f)
    return {}

def sauvegarder_yamazumi(data):
    with open(YAMAZUMI_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    return entrainer_modele(verbose=False)

modele, nom_modele, resultats = charger_modele()

# ─────────────────────────────────────────────
# DONNÉES HISTORIQUES
# ─────────────────────────────────────────────
@st.cache_data
def preparer_historique():
    df = DONNEES_REELLES.copy()
    df['shifts_opt'] = [calculer_shifts_optimises(d,s,r3,r4)[0]
                        for d,s,r3,r4 in zip(df['demande_totale'],df['stock'],
                                              df['demande_ref63'],df['demande_ref64'])]
    df['correction'] = df['shifts_opt'] - df['my_vision']
    df = make_features(df)
    df['shifts_predits'] = np.round(modele.predict(df[FEATURES])).astype(int)
    df['cap_shift'] = df.apply(lambda r: cap_shift_reel(r['demande_ref63'],r['demande_ref64']),axis=1)
    return df

df_hist = preparer_historique()

# ─────────────────────────────────────────────
# DONNÉES DE RÉFÉRENCE YAMAZUMI (W11)
# ─────────────────────────────────────────────
YAMAZUMI_POSTES_REF = [
    {"station": "SPS1",        "ct63": 86.0,  "ct64": 80.0,  "groupe": "SPS"},
    {"station": "OFF LINE1",   "ct63": 66.0,  "ct64": 70.0,  "groupe": "SPS"},
    {"station": "SPS2",        "ct63": 65.0,  "ct64": 80.0,  "groupe": "SPS"},
    {"station": "SPS3",        "ct63": 57.0,  "ct64": 105.0, "groupe": "SPS"},
    {"station": "OFF LINE2",   "ct63": 50.0,  "ct64": 45.0,  "groupe": "SPS"},
    {"station": "SPS4",        "ct63": 90.0,  "ct64": 90.0,  "groupe": "SPS"},
    {"station": "SPS5",        "ct63": 74.0,  "ct64": 80.0,  "groupe": "SPS"},
    {"station": "SPS6",        "ct63": 15.0,  "ct64": 54.0,  "groupe": "SPS"},
    {"station": "SPS7",        "ct63": 15.0,  "ct64": 48.0,  "groupe": "SPS"},
    {"station": "SPS8",        "ct63": 22.0,  "ct64": 54.0,  "groupe": "SPS"},
    {"station": "LAYOUT 01-1", "ct63": 48.0,  "ct64": 70.0,  "groupe": "LAYOUT"},
    {"station": "LAYOUT01",    "ct63": 50.0,  "ct64": 65.0,  "groupe": "LAYOUT"},
    {"station": "LAYOUT02",    "ct63": 50.0,  "ct64": 72.0,  "groupe": "LAYOUT"},
    {"station": "LAYOUT03",    "ct63": 52.0,  "ct64": 70.0,  "groupe": "LAYOUT"},
    {"station": "LAYOUT04",    "ct63": 35.0,  "ct64": 66.0,  "groupe": "LAYOUT"},
    {"station": "LAYOUT05",    "ct63": 41.0,  "ct64": 50.0,  "groupe": "LAYOUT"},
    {"station": "HUMMER",      "ct63": 45.0,  "ct64": 86.0,  "groupe": "LAYOUT"},
    {"station": "/PLUG",       "ct63": 150.0, "ct64": 150.0, "groupe": "TAPPING"},
    {"station": "TAPPING 01",  "ct63": 60.0,  "ct64": 82.0,  "groupe": "TAPPING"},
    {"station": "TAPPING 02",  "ct63": 59.0,  "ct64": 60.0,  "groupe": "TAPPING"},
    {"station": "TAPPING 03",  "ct63": 54.0,  "ct64": 71.0,  "groupe": "TAPPING"},
    {"station": "TAPPING 04",  "ct63": 60.0,  "ct64": 78.0,  "groupe": "TAPPING"},
    {"station": "CLIP 01",     "ct63": 57.0,  "ct64": 69.0,  "groupe": "CLIP"},
    {"station": "CLIP 02",     "ct63": 58.0,  "ct64": 53.0,  "groupe": "CLIP"},
    {"station": "CLIP 03",     "ct63": 56.0,  "ct64": 60.0,  "groupe": "CLIP"},
    {"station": "CLIP 04",     "ct63": 60.0,  "ct64": 66.0,  "groupe": "CLIP"},
    {"station": "REMOVE",      "ct63": 105.0, "ct64": 110.0, "groupe": "CLIP"},
    {"station": "FOAMING 01",  "ct63": 70.0,  "ct64": 80.0,  "groupe": "FOAMING"},
    {"station": "FOAMING 02",  "ct63": 40.0,  "ct64": 49.0,  "groupe": "FOAMING"},
    {"station": "FOAMING 03",  "ct63": 55.0,  "ct64": 73.0,  "groupe": "FOAMING"},
    {"station": "CCB 01",      "ct63": 84.0,  "ct64": 96.0,  "groupe": "CCB"},
    {"station": "CCB 02",      "ct63": 84.0,  "ct64": 96.0,  "groupe": "CCB"},
    {"station": "CCB 03",      "ct63": 84.0,  "ct64": 96.0,  "groupe": "CCB"},
    {"station": "CCB 04",      "ct63": 84.0,  "ct64": 96.0,  "groupe": "CCB"},
    {"station": "CCB 05",      "ct63": 84.0,  "ct64": 96.0,  "groupe": "CCB"},
    {"station": "CCB 06",      "ct63": 84.0,  "ct64": 96.0,  "groupe": "CCB"},
    {"station": "ET 01",       "ct63": 84.0,  "ct64": 96.0,  "groupe": "TEST"},
    {"station": "TV 02",       "ct63": 0.0,   "ct64": 70.0,  "groupe": "TEST"},
    {"station": "VI",          "ct63": 118.0, "ct64": 110.0, "groupe": "FINITION"},
    {"station": "PACK",        "ct63": 125.0, "ct64": 115.0, "groupe": "FINITION"},
]

GROUPE_COULEURS = {
    "SPS":      "#4A9BBF",
    "LAYOUT":   "#2FAE7A",
    "TAPPING": "#9B59B6",
    "CLIP":    "#E67E22",
    "FOAMING": "#16A085",
    "CCB":      "#D94030",
    "TEST":    "#E4A82A",
    "FINITION":"#7F8C8D",
}

TAKT_TIME_YAM   = 75.4   # secondes
CYCLE_IDEAL_YAM = 75.6   # secondes

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏭 Prédiction des Shifts — Ligne NCAR</h1>
  <p>PFE &nbsp;·&nbsp; Optimisation pilotage &amp; équilibrage &nbsp;·&nbsp; Stock sécurité : 1 089 u &nbsp;·&nbsp; Réf63 : 49/h &nbsp;·&nbsp; Réf64 : 33/h (CCB)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres semaine")
    st.markdown("---")
    semaine_num  = st.number_input("Numéro de semaine", 1, 52, 31)
    demande_in   = st.number_input("Demande totale (u/semaine)", 500, 5000, 2200, 50)
    ref63_in     = st.number_input("Demande Réf. 63 (u)", 0, 5000, 1100, 50)
    ref64_in     = st.number_input("Demande Réf. 64 (u)", 0, 5000, 1100, 50)
    stock_in     = st.number_input("Stock actuel (u)", 0, 3000, 80, 10)
    st.markdown("---")
    cap_s = cap_shift_reel(ref63_in, ref64_in)
    cap_d = cap_demie_equipe(ref63_in, ref64_in)
    st.info(f"**Capacité/shift :** {cap_s} u\n\n**Capacité demi-équipe :** {cap_d} u")
    badge = "badge-low" if stock_in < STOCK_SEC else "badge-ok"
    label = "Insuffisant" if stock_in < STOCK_SEC else "Suffisant"
    st.markdown(f'<span class="{badge}">Stock {stock_in}u — {label}</span>', unsafe_allow_html=True)
    st.markdown(f"**Modèle :** {nom_modele}")

# ─────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────
res      = predire_shifts(modele, demande_in, stock_in, ref63_in, ref64_in)
shifts   = res['shifts_optimises']
repart   = repartir_shifts(shifts)
jours_ac = res['jours_actifs']

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📅 Calendrier",
    "📊 Graphiques",
    "📈 Suivi journalier",
    "🔁 Simulation What-If",
    "📋 Historique écarts",
    "📁 Semaines passées",
    "🔩 Ligne NCAR",
    "📦 WIP",
    "📊 Yamazumi Interactif",
    "🤖 Modèle ML",
])

# ══════════════════════════════════════════════
# ONGLET 1 — CALENDRIER
# ══════════════════════════════════════════════
with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><p class="kpi-label">Demande</p><p class="kpi-val" style="color:#2B7FA8;">{demande_in:,}</p><p class="kpi-sub">unités / semaine</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><p class="kpi-label">Shifts optimisés</p><p class="kpi-val" style="color:#2FAE7A;">{shifts}</p><p class="kpi-sub">MY VISION : {res["shifts_my_vision"]}</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><p class="kpi-label">Jours travaillés</p><p class="kpi-val" style="color:#1B6690;">{jours_ac}</p><p class="kpi-sub">{shifts*8}h production</p></div>', unsafe_allow_html=True)
    corr_col = "#B03020" if res['correction']>0 else "#1A6640"
    c4.markdown(f'<div class="kpi-card"><p class="kpi-label">Correction stock</p><p class="kpi-val" style="color:{corr_col};">{res["correction"]:+}</p><p class="kpi-sub">stock {res["stock_statut"]}</p></div>', unsafe_allow_html=True)

    st.markdown(f'<p class="section-title">Planning — Semaine {semaine_num}</p>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i,(jour,col) in enumerate(zip(JOURS_SEMAINE, cols)):
        s = repart[i]
        with col:
            if s == 0:
                st.markdown(f'<div class="shift-repos"><b>{jour}</b><br><small>Repos</small><br><small>—</small><br><b>0h</b></div>', unsafe_allow_html=True)
            elif s == 1:
                st.markdown(f'<div class="shift-matin"><b>{jour}</b><br><small style="color:#185FA5">Shift Matin</small><br><small>07h – 15h</small><br><b>8h</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="shift-matin"><b>{jour}</b><br><small style="color:#185FA5">Shift Matin</small><br><small>07h – 15h</small></div><div class="shift-apm"><small style="color:#1D9E75">Shift Après-midi</small><br><small>15h – 23h</small><br><b>16h total</b></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-title">Résumé équipes</p>', unsafe_allow_html=True)
    r1,r2,r3 = st.columns(3)
    r1.metric("Total shifts semaine", shifts)
    r2.metric("Opérateurs/shift complet", OPE_COMPLET)
    r3.metric("Opérateurs demi-équipe", OPE_DEMI)

# ══════════════════════════════════════════════
# ONGLET 2 — GRAPHIQUES
# ══════════════════════════════════════════════
with tabs[1]:
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    fig.patch.set_facecolor('#EFF6FA')
    fig.suptitle("Analyse des shifts — Ligne NCAR (Semaines 13–30)", fontsize=13, fontweight='bold')

    sems = df_hist['semaine'].tolist()
    dem  = df_hist['demande_totale'].tolist()
    stk  = df_hist['stock'].tolist()
    mv   = df_hist['my_vision'].tolist()
    opt  = df_hist['shifts_opt'].tolist()
    corr = df_hist['correction'].tolist()
    caps = df_hist['cap_shift'].tolist()

    # G1
    ax1 = axes[0,0]; ax1.set_facecolor('#F3F8FB')
    ax1.bar(sems, dem, color='#A8D4E8', alpha=0.75, label='Demande (u)')
    ax1b = ax1.twinx()
    ax1b.plot(sems, mv,  'o--', color='#98B0BE', lw=1.5, ms=6, label='MY VISION')
    ax1b.plot(sems, opt, 'o-',  color='#2B7FA8', lw=2.5, ms=8, label='Shifts optimisés')
    for s,v in zip(sems,opt): ax1b.annotate(str(v),(s,v),textcoords="offset points",xytext=(0,7),ha='center',fontsize=8,color='#2B7FA8',fontweight='bold')
    ax1.set_title('Demande vs Shifts', fontweight='bold')
    ax1.set_xlabel('Semaine'); ax1.set_ylabel('Unités',color='#A8D4E8')
    ax1b.set_ylabel('Shifts',color='#2B7FA8')
    h1=[mpatches.Patch(color='#A8D4E8',label='Demande')]; l1,_=ax1b.get_legend_handles_labels()
    ax1.legend(handles=h1+l1,fontsize=8,loc='upper left'); ax1.grid(axis='y',alpha=0.3)

    # G2
    ax2 = axes[0,1]; ax2.set_facecolor('#F3F8FB')
    cc = ['#D94030' if c>0 else '#2FAE7A' for c in corr]
    ax2.bar(sems, corr, color=cc, alpha=0.85, edgecolor='white')
    ax2.axhline(0,color='black',lw=1)
    for s,c,st_v in zip(sems,corr,stk):
        ax2.annotate(f"stk={int(st_v)}",(s,c),textcoords="offset points",xytext=(0,5 if c>0 else -13),ha='center',fontsize=7,color='gray')
    ax2.set_title('Correction stock appliquée', fontweight='bold')
    ax2.set_xlabel('Semaine'); ax2.set_ylabel('Correction')
    ax2.legend(handles=[mpatches.Patch(color='#D94030',label='Stock<1089 → +1'),
                         mpatches.Patch(color='#2FAE7A',label='Stock≥1089 → -1')],fontsize=8)
    ax2.grid(axis='y',alpha=0.3)

    # G3
    ax3 = axes[1,0]; ax3.set_facecolor('#F3F8FB')
    x=np.arange(len(sems)); w=0.28
    ax3.bar(x-w, mv,  w, color='#A8C0CC', alpha=0.9, label='MY VISION')
    ax3.bar(x,   opt, w, color='#2B7FA8', alpha=0.9, label='Shifts optimisés')
    ax3.bar(x+w, df_hist['shifts_predits'].tolist(), w, color='#2FAE7A', alpha=0.9, label='Prédit ML')
    ax3.set_xticks(x); ax3.set_xticklabels([f'S{s}' for s in sems],rotation=45,fontsize=8)
    ax3.set_title('MY VISION vs Optimisé vs Prédit ML', fontweight='bold')
    ax3.set_ylabel('Shifts'); ax3.legend(fontsize=8); ax3.grid(axis='y',alpha=0.3)

    # G4
    ax4 = axes[1,1]; ax4.set_facecolor('#F3F8FB')
    sc = ax4.scatter(stk, opt, c=['#D94030' if s<STOCK_SEC else '#2FAE7A' for s in stk],
                     s=120, edgecolors='white', lw=1.5, zorder=3)
    ax4.axvline(x=STOCK_SEC,color='#E89020',ls='--',lw=2,label=f'Seuil {STOCK_SEC}u')
    for sv,ov,sem in zip(stk,opt,sems):
        ax4.annotate(f'S{sem}',(sv,ov),textcoords="offset points",xytext=(4,4),fontsize=7,color='gray')
    ax4.set_xlabel('Stock (u)'); ax4.set_ylabel('Shifts optimisés')
    ax4.set_title('Stock ↔ Shifts', fontweight='bold')
    ax4.legend(handles=[mpatches.Patch(color='#D94030',label='Stock insuffisant'),
                         mpatches.Patch(color='#2FAE7A',label='Stock suffisant'),
                         plt.Line2D([0],[0],color='#E89020',ls='--',label=f'Seuil {STOCK_SEC}')],fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Export graphiques
    if st.button("💾 Exporter graphiques (PNG)"):
        fig.savefig('NCAR_graphiques.png', dpi=150, bbox_inches='tight')
        st.success("✅ Fichier 'NCAR_graphiques.png' sauvegardé !")

# ══════════════════════════════════════════════
# ONGLET 3 — SUIVI JOURNALIER
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">Suivi journalier — Objectif vs Réalisé</p>', unsafe_allow_html=True)
    suivi = charger_suivi()
    sem_key = f"S{semaine_num}"
    if sem_key not in suivi:
        suivi[sem_key] = {j: {'objectif': 0, 'realise': 0} for j in JOURS_SEMAINE}

    obj_par_shift = cap_s
    st.info(f"Objectif par shift : **{obj_par_shift} unités** | Capacité demi-équipe : **{cap_d} unités**")

    cols_j = st.columns(6)
    for i,(jour,col) in enumerate(zip(JOURS_SEMAINE, cols_j)):
        s = repart[i]
        obj = s * obj_par_shift
        with col:
            st.markdown(f"**{jour}**")
            st.markdown(f"<small>Objectif : {obj}u</small>", unsafe_allow_html=True)
            real = st.number_input(f"Réalisé", 0, 5000,
                                   value=int(suivi[sem_key][jour].get('realise',0)),
                                   key=f"real_{jour}_{semaine_num}", label_visibility="collapsed")
            suivi[sem_key][jour] = {'objectif': obj, 'realise': real}
            pct = real/obj*100 if obj > 0 else 0
            col_txt = "🟢" if pct >= 95 else "🟡" if pct >= 80 else "🔴"
            st.markdown(f"{col_txt} {pct:.0f}%")

    if st.button("💾 Sauvegarder suivi"):
        sauvegarder_suivi(suivi)
        st.success("✅ Suivi sauvegardé !")

    # Graphique objectif vs réalisé
    objs  = [suivi[sem_key][j]['objectif'] for j in JOURS_SEMAINE]
    reals = [suivi[sem_key][j]['realise']  for j in JOURS_SEMAINE]

    if any(r > 0 for r in reals):
        fig2, ax = plt.subplots(figsize=(10,4))
        x = np.arange(6); w = 0.35
        ax.bar(x-w/2, objs,  w, color='#A8D4E8', alpha=0.85, label='Objectif')
        ax.bar(x+w/2, reals, w, color='#2B7FA8', alpha=0.85, label='Réalisé')
        # Ligne cumul
        cum_obj  = np.cumsum(objs)
        cum_real = np.cumsum(reals)
        ax2b = ax.twinx()
        ax2b.plot(x, cum_obj,  's--', color='#98B0BE', lw=1.5, ms=6, label='Cumul objectif')
        ax2b.plot(x, cum_real, 'o-',  color='#D94030', lw=2,   ms=8, label='Cumul réalisé')
        ax.set_xticks(x); ax.set_xticklabels(JOURS_SEMAINE)
        ax.set_ylabel('Unités/jour'); ax2b.set_ylabel('Cumul semaine')
        ax.set_title(f'Suivi journalier — Semaine {semaine_num}', fontweight='bold')
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2b.get_legend_handles_labels()
        ax.legend(handles=h1+h2, labels=l1+l2, fontsize=9, loc='upper left')
        ax.grid(axis='y',alpha=0.3)
        st.pyplot(fig2)

        # Alerte retard
        ecart_cum = cum_real[-1] - cum_obj[-1]
        if ecart_cum < -obj_par_shift * 0.5:
            st.error(f"⚠️ Retard cumulé de {abs(int(ecart_cum))} unités — Envisager un shift supplémentaire !")
        elif ecart_cum >= 0:
            st.success(f"✅ Production en avance de {int(ecart_cum)} unités sur l'objectif !")

# ══════════════════════════════════════════════
# ONGLET 4 — SIMULATION WHAT-IF
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title">Simulation What-If — Impact sur les shifts</p>', unsafe_allow_html=True)
    st.info("Modifiez les paramètres ci-dessous pour simuler différents scénarios.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scénario de base**")
        st.write(f"Demande : {demande_in:,} u | Stock : {stock_in} u | Shifts : {shifts}")
    with col2:
        st.markdown("**Modifier le scénario**")
        dem_sim   = st.slider("Demande (+/- %)", -50, 50, 0, 5)
        stk_sim   = st.slider("Stock (+/- unités)", -500, 500, 0, 50)
        r63_sim   = st.slider("Ratio Réf.63 (%)", 20, 80, int(ref63_in/(ref63_in+ref64_in+1)*100), 5)

    dem_new  = int(demande_in * (1 + dem_sim/100))
    stk_new  = max(0, stock_in + stk_sim)
    ref63_new = int(dem_new * r63_sim / 100)
    ref64_new = dem_new - ref63_new

    res_sim = predire_shifts(modele, dem_new, stk_new, ref63_new, ref64_new)
    cap_sim = cap_shift_reel(ref63_new, ref64_new)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    delta_sh = res_sim['shifts_optimises'] - shifts
    c1.metric("Demande simulée", f"{dem_new:,} u", f"{dem_sim:+}%")
    c2.metric("Stock simulé", f"{stk_new} u", f"{stk_sim:+}")
    c3.metric("Cap/shift simulée", f"{cap_sim} u")
    c4.metric("Shifts simulés", res_sim['shifts_optimises'], f"{delta_sh:+}")

    if delta_sh > 0:
        st.error(f"⚠️ Ce scénario nécessite **{delta_sh} shift(s) supplémentaire(s)** par rapport au plan actuel.")
    elif delta_sh < 0:
        st.success(f"✅ Ce scénario permet d'**économiser {abs(delta_sh)} shift(s)** par rapport au plan actuel.")
    else:
        st.info("✅ Ce scénario ne modifie pas le nombre de shifts.")

    # Comparaison répartition
    repart_sim = repartir_shifts(res_sim['shifts_optimises'])
    st.markdown('<p class="section-title">Répartition simulée</p>', unsafe_allow_html=True)
    cols_s = st.columns(6)
    for i,(jour,col) in enumerate(zip(JOURS_SEMAINE,cols_s)):
        s_base, s_sim = repart[i], repart_sim[i]
        with col:
            delta = s_sim - s_base
            emoji = "🔺" if delta > 0 else "🔻" if delta < 0 else "✅"
            st.markdown(f"**{jour}**")
            st.markdown(f"Base : {s_base} sh → Sim : {s_sim} sh {emoji}")

# ══════════════════════════════════════════════
# ONGLET 5 — HISTORIQUE ÉCARTS
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-title">Historique des écarts — MY VISION vs Optimisé vs Prédit ML</p>', unsafe_allow_html=True)

    df_aff = df_hist[['semaine','demande_totale','stock','cap_shift',
                       'my_vision','shifts_opt','shifts_predits','correction']].copy()
    df_aff.columns = ['Semaine','Demande','Stock','Cap/shift',
                       'MY VISION','Shifts opt.','Prédit ML','Correction']
    df_aff['Statut ML'] = df_aff.apply(
        lambda r: '✅ Correct' if r['Shifts opt.']==r['Prédit ML'] else '⚠️ Écart', axis=1)
    df_aff['Analyse'] = df_aff.apply(
        lambda r: 'Stock toujours < 1089 → +1 systématique' if r['Correction']==1
                  else 'Stock ≥ 1089 → -1 possible' if r['Correction']==-1
                  else 'Pas de correction', axis=1)

    st.dataframe(df_aff.style.map(
        lambda v: 'color:#C0392B;font-weight:600' if v=='⚠️ Écart'
                  else 'color:#1B5E20;font-weight:600' if v=='✅ Correct' else '',
        subset=['Statut ML']), use_container_width=True, hide_index=True)
    
    exact = (df_hist['shifts_opt']==df_hist['shifts_predits']).sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("Prédictions exactes", f"{exact}/18 ({int(exact/18*100)}%)")
    c2.metric("MAE", f"{abs(df_hist['shifts_opt']-df_hist['shifts_predits']).mean():.3f} shifts")
    c3.metric("Semaines +1 shift (stock)", f"{(df_hist['correction']==1).sum()}/18")

    # Export Excel
    if st.button("📥 Exporter historique Excel"):
        df_aff.to_excel('NCAR_historique.xlsx', index=False)
        st.success("✅ Fichier 'NCAR_historique.xlsx' sauvegardé !")

# ══════════════════════════════════════════════
# ONGLET 6 — SEMAINES PASSÉES
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-title">Archive semaines passées</p>', unsafe_allow_html=True)
    suivi_all = charger_suivi()

    if suivi_all:
        for sem, jours_data in sorted(suivi_all.items()):
            if sem.startswith("WIP_") or sem.startswith("YAM_"):
                continue  # Éviter de lister les données WIP ou Yamazumi ici
            with st.expander(f"📅 {sem}"):
                rows = []
                for jour, vals in jours_data.items():
                    obj  = vals.get('objectif', 0)
                    real = vals.get('realise', 0)
                    pct  = real/obj*100 if obj > 0 else 0
                    rows.append({'Jour':jour,'Objectif':obj,'Réalisé':real,'Taux (%)':round(pct,1)})
                df_sem = pd.DataFrame(rows)
                st.dataframe(df_sem, use_container_width=True, hide_index=True)
                total_obj  = df_sem['Objectif'].sum()
                total_real = df_sem['Réalisé'].sum()
                taux_global = total_real/total_obj*100 if total_obj>0 else 0
                st.metric("Taux global semaine", f"{taux_global:.1f}%",
                          f"{total_real-total_obj:+} u vs objectif")
    else:
        st.info("Aucune donnée de suivi enregistrée pour le moment. Utilisez l'onglet 'Suivi journalier'.")

    if st.button("📥 Exporter tout l'historique Excel"):
        rows_all = []
        for sem, jours_data in sorted(suivi_all.items()):
            if sem.startswith("WIP_") or sem.startswith("YAM_"):
                continue
            for jour, vals in jours_data.items():
                rows_all.append({'Semaine':sem,'Jour':jour,
                                  'Objectif':vals.get('objectif',0),
                                  'Réalisé':vals.get('realise',0)})
        pd.DataFrame(rows_all).to_excel('NCAR_historique_complet.xlsx', index=False)
        st.success("✅ 'NCAR_historique_complet.xlsx' sauvegardé !")

# ══════════════════════════════════════════════
# ONGLET 7 — LIGNE NCAR
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-title">Équilibrage de la ligne NCAR — Yamazumi static</p>', unsafe_allow_html=True)

    TAKT_TIME   = 75.4  # secondes
    CYCLE_IDEAL = 75.6  # secondes (1.26 min)

    postes = {
        'SPS1':    {'temps': 62, 'goulot': False},
        'SPS2':    {'temps': 68, 'goulot': False},
        'LAYOUT':  {'temps': 70, 'goulot': False},
        'FOAMING': {'temps': 71, 'goulot': False},
        'TE':      {'temps': 78, 'goulot': True},
        'TV':      {'temps': 80, 'goulot': True},
        'CCB':     {'temps': 112,'goulot': True},
    }

    fig3, ax = plt.subplots(figsize=(12,6))
    ax.set_facecolor('#F3F8FB')
    noms  = list(postes.keys())
    temps = [postes[p]['temps'] for p in noms]
    cols_p= ['#D94030' if postes[p]['goulot'] else '#2B7FA8' for p in noms]

    bars = ax.bar(noms, temps, color=cols_p, alpha=0.85, edgecolor='white', lw=1.5, width=0.6)
    ax.axhline(y=TAKT_TIME,  color='#E89020', ls='--', lw=2.5, label=f'Takt Time ({TAKT_TIME}s)')
    ax.axhline(y=CYCLE_IDEAL,color='#2FAE7A', ls=':',  lw=2,   label=f'Cycle idéal ({CYCLE_IDEAL}s)')

    for bar, t, nom in zip(bars, temps, noms):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{t}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if postes[nom]['goulot']:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    '⚠️', ha='center', va='center', fontsize=14)

    ax.set_ylabel('Temps de cycle (secondes)', fontsize=11)
    ax.set_title('Yamazumi Chart static — Ligne NCAR\nRouge = postes goulots (TE, TV, CCB)', fontweight='bold', fontsize=12)
    p1 = mpatches.Patch(color='#2B7FA8', label='Poste normal')
    p2 = mpatches.Patch(color='#D94030', label='Poste goulot (> Takt Time)')
    ax.legend(handles=[p1,p2,
              plt.Line2D([0],[0],color='#E89020',ls='--',lw=2,label=f'Takt Time ({TAKT_TIME}s)'),
              plt.Line2D([0],[0],color='#2FAE7A',ls=':',lw=2,label=f'Cycle idéal ({CYCLE_IDEAL}s)')],
              fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 130)
    st.pyplot(fig3)

    st.markdown('<p class="section-title">Analyse des goulots</p>', unsafe_allow_html=True)
    for nom, data in postes.items():
        if data['goulot']:
            ecart = data['temps'] - TAKT_TIME
            st.markdown(f"""<div class="goulot-card">
                <b>⚠️ {nom}</b> — Temps cycle : {data['temps']}s |
                Dépassement Takt Time : <b style="color:#E53935">+{ecart:.1f}s</b><br>
                <small>Impact : ralentit toute la ligne · CCB limite la Réf.64 à 33 câbles/h</small>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Impact shifts sur la ligne</p>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Shifts prédits cette semaine", shifts)
    c2.metric("Production attendue", f"{shifts * cap_s:,} u")
    c3.metric("Pression sur CCB", f"{shifts * 8 * 33:,} u Réf.64" if ref64_in > 0 else "—")

    if shifts >= 8:
        st.warning(f"⚠️ Avec {shifts} shifts, le poste CCB tourne à pleine capacité. Surveiller les arrêts et la qualité.")

    if st.button("📥 Exporter Yamazumi static (PNG)"):
        fig3.savefig('NCAR_yamazumi.png', dpi=150, bbox_inches='tight')
        st.success("✅ 'NCAR_yamazumi.png' sauvegardé !")

# ══════════════════════════════════════════════
# ONGLET 8 — WIP (Encours de production)
# ══════════════════════════════════════════════
with tabs[7]:
    SEUIL_WIP = 8

    POSTES_WIP = [
        {'nom': 'SPS1',    'couleur': '#4A9BBF'},
        {'nom': 'SPS2',    'couleur': '#4A9BBF'},
        {'nom': 'LAYOUT',  'couleur': '#4A9BBF'},
        {'nom': 'FOAMING', 'couleur': '#4A9BBF'},
        {'nom': 'AS1',     'couleur': '#4A9BBF'},
        {'nom': 'AS2',     'couleur': '#4A9BBF'},
        {'nom': 'AS3',     'couleur': '#4A9BBF'},
        {'nom': 'AS4',     'couleur': '#4A9BBF'},
        {'nom': 'PRE',     'couleur': '#e67e22'},
        {'nom': 'CLP',     'couleur': '#f5a623'},
        {'nom': 'TE',      'couleur': '#e74c3c'},
        {'nom': 'TV',      'couleur': '#f5a623'},
        {'nom': 'CCB',     'couleur': '#f5a623'},
        {'nom': 'EOL',     'couleur': '#27ae60'},
        {'nom': 'VI',      'couleur': '#ffffff'},
        {'nom': 'PACK',    'couleur': '#ffffff'},
        {'nom': 'FW',      'couleur': '#4A9BBF'},
    ]

    st.markdown('<p class="section-title">📦 WIP — Encours avant chaque poste</p>', unsafe_allow_html=True)
    st.info(f"Seuil d'alerte : **{SEUIL_WIP} pièces**. Au-dessus de ce seuil, le poste est signalé en rouge.")

    # Schéma ligne NCAR inséré directement depuis votre code HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manufacturing Concept Template - YAZAKI</title>
        <style>
            :root {
                --bg-color: #ffffff;
                --border-color: #333;
                --box-blue: #4a90e2;
                --box-yellow: #f5a623;
                --box-black: #000000;
                --box-green: #27ae60;
                --box-orange: #e67e22;
                --op-green: #2ecc71;
                --op-red: #e74c3c;
                --op-yellow: #f1c40f;
                --text-color: #333;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
            }

            .manufacturing-container {
                background: var(--bg-color);
                width: 100%;
                max-width: 1400px;
                border: 2px solid var(--border-color);
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }

            /* --- HEADER --- */
            .header {
                display: grid;
                grid-template-columns: 1fr 2fr 1fr;
                border-bottom: 2px solid var(--border-color);
                padding: 10px;
                background: #fff;
            }

            .header-left, .header-right {
                display: flex;
                flex-direction: column;
                justify-content: center;
                font-size: 0.8rem;
                border: 1px solid #ccc;
                margin: 0 5px;
                padding: 5px;
            }

            .header-center {
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }

            .header-center h1 {
                margin: 0;
                font-size: 1.5rem;
                color: #555;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .grid-info {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 5px;
                width: 100%;
                margin-top: 5px;
                font-size: 0.75rem;
            }

            .info-box {
                background: #eef2f5;
                border: 1px solid #ddd;
                padding: 2px;
                text-align: center;
            }
            .info-label { font-weight: bold; color: #555; }
            .info-val { color: #000; }

            /* --- MAIN CONTENT --- */
            .content-wrapper {
                display: flex;
                padding: 20px;
                gap: 20px;
            }

            .sidebar-left {
                width: 70px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                align-items: center;
            }
            .fez-box {
                width: 100%;
                height: 120px;
                background: #ffccbc;
                border: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: #d84315;
                writing-mode: vertical-rl;
                transform: rotate(180deg);
                font-size: 1rem;
            }
            .small-box {
                width: 40px;
                height: 40px;
                border: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.7rem;
                font-weight: bold;
            }

            .main-schema {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }

            /* Ligne du haut */
            .top-row {
                display: flex;
                justify-content: space-around;
                align-items: center;
                border-bottom: 1px dashed #ccc;
                padding-bottom: 10px;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .station {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 5px;
                min-width: 55px;
            }

            .station-box {
                padding: 5px 10px;
                border: 1px solid #333;
                font-size: 0.7rem;
                font-weight: bold;
                text-align: center;
                min-width: 40px;
            }
            .bg-blue { background: var(--box-blue); color: white; }
            .bg-yellow { background: var(--box-yellow); color: white; }
            .bg-black { background: var(--box-black); color: white; }
            .bg-green { background: var(--box-green); color: white; }
            .bg-orange { background: var(--box-orange); color: white; }
            .bg-white { background: white; }

            .operator-icon {
                width: 30px;
                height: 20px;
                border-radius: 10px;
                border: 1px solid #333;
            }
            .icon-green { background: var(--op-green); }
            .icon-red { background: var(--op-red); }
            .icon-yellow { background: var(--op-yellow); }
            .icon-blue { background: var(--box-blue); }

            /* Process Flow */
            .process-flow {
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 10px;
                align-items: center;
            }

            .process-step {
                border: 1px solid #ddd;
                padding: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 5px;
                background: #fafafa;
                position: relative;
            }

            .step-number {
                font-size: 0.7rem;
                color: #777;
                align-self: flex-start;
            }

            .box-label {
                background: var(--box-blue);
                color: white;
                padding: 2px 8px;
                font-size: 0.7rem;
                width: 80%;
                text-align: center;
            }

            /* NOUVELLE LIGNE : Postes additionnels */
            .additional-row {
                display: grid;
                grid-template-columns: repeat(8, 1fr);
                gap: 8px;
                border-top: 1px dashed #ccc;
                padding-top: 10px;
                margin-top: 5px;
            }

            .add-station {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 4px;
                padding: 5px;
                border: 1px solid #eee;
                background: #fafbfc;
            }

            /* Sidebar Droite */
            .sidebar-right {
                width: 250px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                border-left: 1px solid #eee;
            }
            
            .conveyor-placeholder {
                width: 100%;
                height: 200px;
                background: #e0e0e0;
                border: 2px dashed #999;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
                font-style: italic;
                position: relative;
            }
            
            .conveyor-circle {
                width: 150px;
                height: 150px;
                border: 8px solid #ccc;
                border-radius: 50%;
                position: relative;
                background: #fff;
            }
            .conveyor-circle::after {
                content: '';
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                width: 10px; height: 10px;
                background: #333;
                border-radius: 50%;
            }

            /* LEGEND */
            .legend-section {
                border-top: 2px solid var(--border-color);
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                background: #f9f9f9;
                flex-wrap: wrap;
                gap: 15px;
            }

            .legend-items {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 0.8rem;
            }

            .legend-key {
                width: 15px;
                height: 10px;
                border-radius: 4px;
                border: 1px solid #333;
            }

            .bottom-bar {
                display: flex;
                gap: 5px;
                margin-top: 10px;
                flex-wrap: wrap;
            }
            .bar-item {
                padding: 4px 8px;
                font-size: 0.7rem;
                border: 1px solid #333;
                background: #eee;
            }

            @media (max-width: 1024px) {
                .content-wrapper { flex-direction: column; }
                .sidebar-right { width: 100%; border-left: none; border-top: 1px solid #eee; padding-top: 20px;}
                .process-flow { grid-template-columns: repeat(4, 1fr); }
                .additional-row { grid-template-columns: repeat(4, 1fr); }
                .header { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>

        <div class="manufacturing-container">
            <header class="header">
                <div class="header-left">
                    <div class="info-box"><span class="info-label">PLANT</span><br><span class="info-val">YMM2</span></div>
                </div>
                <header class="header-center">
                    <h1>Manufacturing Concept Template</h1>
                    <div class="grid-info">
                        <div class="info-box"><span class="info-label">Project</span><br><span class="info-val">BMW NCAR</span></div>
                        <div class="info-box"><span class="info-label">Customer</span><br><span class="info-val">BMW</span></div>
                        <div class="info-box"><span class="info-label">Phase</span><br><span class="info-val">VS1</span></div>
                        <div class="info-box"><span class="info-label">HECKEND</span><br><span class="info-val">-</span></div>
                    </div>
                </header>
                <div class="header-right">
                    <div style="font-size: 0.7rem; margin-bottom:5px;">
                        <strong>Signature:</strong> Souha . Amslek <br>
                        <strong>Revision Date:</strong> 21/05/2025
                    </div>
                </div>
            </header>

            <div class="content-wrapper">
                
                <aside class="sidebar-left">
                    <div class="fez-box">FEZ</div>
                    <div class="small-box">IN</div>
                    <div class="small-box bg-green" style="color:white">OK</div>
                </aside>

                <main class="main-schema">
                    
                    <div class="top-row">
                        <div class="station">
                            <div class="station-box bg-blue">FW</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-white">PACK</div>
                            <div class="operator-icon icon-yellow"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-white">VI</div>
                            <div class="operator-icon icon-red"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-yellow">TV</div>
                            <div class="operator-icon icon-yellow"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-yellow">CCB</div>
                            <div class="operator-icon icon-yellow"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-green">EOL</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-orange">REW</div>
                            <div class="operator-icon icon-red"></div>
                        </div>
                        <div class="station">
                            <div class="station-box bg-black">SPS</div>
                            <div class="operator-icon icon-yellow"></div>
                        </div>
                    </div>

                    <div class="process-flow">
                        <div class="process-step">
                            <span class="step-number">7 / 8</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-red"></div>
                        </div>
                        <div class="process-step">
                            <span class="step-number">9 / 6</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="process-step">
                            <span class="step-number">10 / 5</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="process-step">
                            <span class="step-number">11 / 4</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="process-step">
                            <span class="step-number">12 / 3</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="process-step">
                            <span class="step-number">13 / 2</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                        <div class="process-step">
                            <span class="step-number">14 / 1</span>
                            <div class="box-label">BOX</div>
                            <div class="operator-icon icon-green"></div>
                        </div>
                    </div>

                    <div class="additional-row">
                        <div class="add-station">
                            <div class="station-box bg-blue">AS1</div>
                            <div class="operator-icon icon-green"></div>
                            <span style="font-size:0.65rem;">Assembly 1</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-blue">AS2</div>
                            <div class="operator-icon icon-green"></div>
                            <span style="font-size:0.65rem;">Assembly 2</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-blue">AS3</div>
                            <div class="operator-icon icon-yellow"></div>
                            <span style="font-size:0.65rem;">Assembly 3</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-blue">AS4</div>
                            <div class="operator-icon icon-green"></div>
                            <span style="font-size:0.65rem;">Assembly 4</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-orange">PRE</div>
                            <div class="operator-icon icon-red"></div>
                            <span style="font-size:0.65rem;">Pre-Assy</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-yellow">CLP</div>
                            <div class="operator-icon icon-yellow"></div>
                            <span style="font-size:0.65rem;">Clipping</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-green">QC</div>
                            <div class="operator-icon icon-green"></div>
                            <span style="font-size:0.65rem;">Quality</span>
                        </div>
                        <div class="add-station">
                            <div class="station-box bg-black">FIN</div>
                            <div class="operator-icon icon-blue"></div>
                            <span style="font-size:0.65rem;">Finish</span>
                        </div>
                    </div>

                </main>

                <aside class="sidebar-right">
                    <div class="conveyor-placeholder">
                        <div class="conveyor-circle"></div>
                        <span style="position:absolute; bottom:10px; right:10px; font-size:0.7rem;">3D Conveyor View</span>
                    </div>
                </aside>
            </div>

            <footer class="legend-section">
                <div class="legend-items">
                    <div class="legend-item"><div class="legend-key" style="background:#2ecc71"></div> SPS operator</div>
                    <div class="legend-item"><div class="legend-key" style="background:#e74c3c"></div> Taping operator</div>
                    <div class="legend-item"><div class="legend-key" style="background:#3498db"></div> Direct assembly</div>
                    <div class="legend-item"><div class="legend-key" style="background:#f1c40f"></div> Testing & Pack operator</div>
                    <div class="legend-item"><div class="legend-key" style="background:#e67e22"></div> Pre-Assembly</div>
                    <div class="legend-item"><div class="legend-key" style="background:#27ae60"></div> Quality / EOL</div>
                </div>
                
                <div class="bottom-bar">
                    <div class="bar-item" style="background:#2ecc71; color:white">PLUG WS</div>
                    <div class="bar-item" style="background:#f1c40f">DS20</div>
                    <div class="bar-item" style="background:#f1c40f">DS20</div>
                    <div class="bar-item" style="background:#000; color:white">SPS</div>
                    <div class="bar-item" style="background:#000; color:white">SPS</div>
                    <div class="bar-item" style="background:#3498db; color:white">SPS6</div>
                    <div class="bar-item" style="background:#3498db; color:white">QFL3</div>
                    <div class="bar-item" style="background:#3498db; color:white">SPS5</div>
                    <div class="bar-item" style="background:#3498db; color:white">SPS4</div>
                    <div class="bar-item" style="background:#3498db; color:white">QFL2</div>
                    <div class="bar-item" style="background:#3498db; color:white">SPS3</div>
                    <div class="bar-item" style="background:#3498db; color:white">SPS2</div>
                    <div class="bar-item" style="background:#3498db; color:white">QFL1</div>
                    <div class="bar-item" style="background:#3498db; color:white">SPS1</div>
                    <div class="bar-item" style="background:#3498db; color:white">KANBAN</div>
                </div>
            </footer>
        </div>

    </body>
    </html>
    """
    st.components.v1.html(html_content, height=550, scrolling=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Saisie des encours par poste</p>', unsafe_allow_html=True)

    # Chargement WIP sauvegardé
    wip_key = f"WIP_S{semaine_num}"
    suivi_wip = charger_suivi()
    if wip_key not in suivi_wip:
        suivi_wip[wip_key] = {p['nom']: 0 for p in POSTES_WIP}

    # Grille de saisie — 6 colonnes
    cols_wip = st.columns(6)
    wip_vals = {}
    for i, poste in enumerate(POSTES_WIP):
        with cols_wip[i % 6]:
            val = st.number_input(
                poste['nom'], min_value=0, max_value=100,
                value=int(suivi_wip[wip_key].get(poste['nom'], 0)),
                key=f"wip_{poste['nom']}_{semaine_num}"
            )
            wip_vals[poste['nom']] = val
            alerte = val >= SEUIL_WIP
            if alerte:
                st.markdown(f"<span style='color:#E53935;font-size:11px;font-weight:600;'>⚠️ {val} ≥ {SEUIL_WIP}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#1B5E20;font-size:11px;'>✅ {val}</span>", unsafe_allow_html=True)

    if st.button("💾 Sauvegarder WIP"):
        suivi_wip[wip_key] = wip_vals
        sauvegarder_suivi(suivi_wip)
        st.success("✅ Encours sauvegardés !")

    # Graphique barres WIP
    st.markdown('<p class="section-title">Visualisation des encours</p>', unsafe_allow_html=True)
    fig_wip, ax_wip = plt.subplots(figsize=(14, 5))
    ax_wip.set_facecolor('#F3F8FB')
    noms_p  = [p['nom'] for p in POSTES_WIP]
    vals_p  = [wip_vals.get(p['nom'], 0) for p in POSTES_WIP]
    cols_p  = ['#D94030' if v >= SEUIL_WIP else '#2B7FA8' for v in vals_p]

    bars_wip = ax_wip.bar(noms_p, vals_p, color=cols_p, alpha=0.85, edgecolor='white', lw=1.5, width=0.6)
    ax_wip.axhline(y=SEUIL_WIP, color='#E89020', ls='--', lw=2.5, label=f"Seuil alerte ({SEUIL_WIP} pièces)")

    for bar, v in zip(bars_wip, vals_p):
        ax_wip.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(v), ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#D94030' if v >= SEUIL_WIP else '#374151')

    ax_wip.set_ylabel("Nombre de pièces en encours", fontsize=11)
    ax_wip.set_title(f"WIP avant chaque poste — Semaine {semaine_num}\n(Rouge = encours ≥ seuil d'alerte {SEUIL_WIP})", fontweight='bold')
    ax_wip.legend(fontsize=10)
    ax_wip.set_ylim(0, max(max(vals_p) + 3, SEUIL_WIP + 3))
    ax_wip.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_wip)

    # Résumé alertes
    postes_alertes = [p for p in noms_p if wip_vals.get(p, 0) >= SEUIL_WIP]
    if postes_alertes:
        st.error(f"⚠️ **{len(postes_alertes)} poste(s) en alerte :** {', '.join(postes_alertes)}")
        st.markdown("**Action recommandée :** Vérifier les causes de blocage (panne, manque opérateur, qualité) sur ces postes.")
    else:
        st.success("✅ Tous les postes sont sous le seuil d'alerte.")

    if st.button("📥 Exporter WIP Excel"):
        df_wip = pd.DataFrame([{'Poste': p, 'Encours': v,
                                  'Statut': '⚠️ Alerte' if v >= SEUIL_WIP else '✅ OK'}
                                 for p, v in wip_vals.items()])
        df_wip.to_excel('NCAR_WIP.xlsx', index=False)
        st.success("✅ 'NCAR_WIP.xlsx' sauvegardé !")

# ══════════════════════════════════════════════
# ONGLET 9 — YAMAZUMI INTERACTIF
# ══════════════════════════════════════════════
with tabs[8]:

    st.markdown('<p class="section-title">📊 Yamazumi Chart — Saisie & Analyse des temps de cycle</p>', unsafe_allow_html=True)

    # ── Paramètres globaux ─────────────────────
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        sem_yam = st.number_input("Semaine Yamazumi", 1, 52,
                                   value=semaine_num, key="yam_semaine")
    with col_p2:
        takt_yam = st.number_input("Takt Time (s)", 30.0, 300.0,
                                    value=float(TAKT_TIME_YAM), step=0.5, key="yam_takt")
    with col_p3:
        ref_filter = st.selectbox("Référence à afficher",
                                   ["Moyenne", "Réf. 63", "Réf. 64"],
                                   key="yam_ref")

    st.markdown("---")

    # ── Chargement données sauvegardées ────────
    yam_data = charger_yamazumi()
    yam_key  = f"YAM_S{sem_yam}"

    if yam_key not in yam_data:
        # Initialiser avec valeurs de référence W11
        yam_data[yam_key] = {
            p["station"]: {"ct63": p["ct63"] or 0.0, "ct64": p["ct64"] or 0.0}
            for p in YAMAZUMI_POSTES_REF
        }

    saved = yam_data[yam_key]

    # ── Saisie par groupe de postes ─────────────
    st.markdown('<p class="section-title">✏️ Saisie des temps de cycle par poste (secondes)</p>', unsafe_allow_html=True)

    groupes_uniques = list(dict.fromkeys(p["groupe"] for p in YAMAZUMI_POSTES_REF))
    ct_values = {}   # station → {ct63, ct64}

    for grp in groupes_uniques:
        postes_grp = [p for p in YAMAZUMI_POSTES_REF if p["groupe"] == grp]
        couleur_grp = GROUPE_COULEURS.get(grp, "#888")

        st.markdown(
            f'<div style="background:{couleur_grp}22;border-left:4px solid {couleur_grp};'
            f'padding:6px 12px;border-radius:6px;margin-bottom:4px;">'
            f'<b style="color:{couleur_grp};">{grp}</b></div>',
            unsafe_allow_html=True
)

        n_cols = min(len(postes_grp), 5)
        grp_cols = st.columns(n_cols)

        for i, poste in enumerate(postes_grp):
            nom = poste["station"]
            prev63 = float(saved.get(nom, {}).get("ct63") or 0)
            prev64 = float(saved.get(nom, {}).get("ct64") or 0)

            with grp_cols[i % n_cols]:
                st.markdown(f"**{nom}**")
                v63 = st.number_input(f"CT63 (s)", 0.0, 500.0,
                                      value=prev63, step=1.0,
                                      key=f"yam_{nom}_63_{sem_yam}",
                                      label_visibility="visible")
                v64 = st.number_input(f"CT64 (s)", 0.0, 500.0,
                                      value=prev64, step=1.0,
                                      key=f"yam_{nom}_64_{sem_yam}",
                                      label_visibility="visible")
                ct_values[nom] = {"ct63": v63, "ct64": v64}

                # Mini alerte inline
                avg_val = (v63 + v64) / 2 if (v63 + v64) > 0 else 0
                ref_val = v63 if ref_filter == "Réf. 63" else v64 if ref_filter == "Réf. 64" else avg_val
                if ref_val > takt_yam:
                    st.markdown(f"<span style='color:#D94030;font-size:10px;'>⚠️ {ref_val:.0f}s > TT</span>",
                                unsafe_allow_html=True)
                elif ref_val > 0:
                    st.markdown(f"<span style='color:#1A6640;font-size:10px;'>✅ {ref_val:.0f}s</span>",
                                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Bouton sauvegarde ──────────────────────
    col_sv1, col_sv2 = st.columns([1, 4])
    with col_sv1:
        if st.button("💾 Sauvegarder Yamazumi", key="save_yam"):
            yam_data[yam_key] = ct_values
            sauvegarder_yamazumi(yam_data)
            st.success(f"✅ Données Yamazumi S{sem_yam} sauvegardées !")

    st.markdown("---")

    # ── Construction du graphique Yamazumi ─────
    st.markdown('<p class="section-title">📈 Yamazumi Chart — Visualisation</p>', unsafe_allow_html=True)

    # Choisir la valeur à afficher selon le filtre
    noms_affich, vals_affich, couleurs_affich, groupes_affich = [], [], [], []
    for poste in YAMAZUMI_POSTES_REF:
        nom  = poste["station"]
        grp  = poste["groupe"]
        v63  = ct_values.get(nom, {}).get("ct63", 0) or 0
        v64  = ct_values.get(nom, {}).get("ct64", 0) or 0
        avg  = (v63 + v64) / 2 if (v63 + v64) > 0 else 0

        if ref_filter == "Réf. 63":
            val = v63
        elif ref_filter == "Réf. 64":
            val = v64
        else:
            val = avg

        if val > 0:
            noms_affich.append(nom)
            vals_affich.append(val)
            groupes_affich.append(grp)
            base_col = GROUPE_COULEURS.get(grp, "#888")
            # Rouge si dépasse takt, sinon couleur groupe
            couleurs_affich.append("#D94030" if val > takt_yam else base_col)

    if not vals_affich:
        st.info("Aucun temps de cycle saisi. Remplissez les champs ci-dessus.")
    else:
        fig_yam, ax_yam = plt.subplots(figsize=(max(14, len(noms_affich) * 0.45), 7))
        fig_yam.patch.set_facecolor('#EFF6FA')
        ax_yam.set_facecolor('#F3F8FB')

        x_pos  = np.arange(len(noms_affich))
        bars_y = ax_yam.bar(x_pos, vals_affich, color=couleurs_affich,
                            alpha=0.87, edgecolor='white', lw=1.2, width=0.75)

        # Lignes de référence
        ax_yam.axhline(y=takt_yam, color='#E89020', ls='--', lw=2.5,
                       label=f'Takt Time ({takt_yam}s)')
        ax_yam.axhline(y=CYCLE_IDEAL_YAM, color='#2FAE7A', ls=':', lw=2,
                       label=f'Cycle idéal ({CYCLE_IDEAL_YAM}s)')

        # Étiquettes valeurs
        for bar_b, v, nom in zip(bars_y, vals_affich, noms_affich):
            ax_yam.text(
                bar_b.get_x() + bar_b.get_width() / 2,
                bar_b.get_height() + 1.2,
                f'{v:.0f}s',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color='#D94030' if v > takt_yam else '#1B3A4B'
            )
            if v > takt_yam:
                ax_yam.text(
                    bar_b.get_x() + bar_b.get_width() / 2,
                    bar_b.get_height() / 2,
                    '⚠️', ha='center', va='center', fontsize=11
                )

        # Axe X — groupes colorés
        ax_yam.set_xticks(x_pos)
        ax_yam.set_xticklabels(noms_affich, rotation=45, ha='right', fontsize=8)

        # Colorier chaque label selon son groupe
        for tick_label, grp in zip(ax_yam.get_xticklabels(), groupes_affich):
            tick_label.set_color(GROUPE_COULEURS.get(grp, "#333"))

        ax_yam.set_ylabel('Temps de cycle (s)', fontsize=11)
        ax_yam.set_title(
            f'Yamazumi Chart — Ligne NCAR | Semaine {sem_yam} | {ref_filter}\n'
            f'Rouge = postes > Takt Time ({takt_yam}s)',
            fontweight='bold', fontsize=12
        )
        ax_yam.set_ylim(0, max(max(vals_affich) * 1.15, takt_yam * 1.3))
        ax_yam.grid(axis='y', alpha=0.3)

        # Légende groupes + lignes
        legend_patches = [
            mpatches.Patch(color=c, label=g, alpha=0.85)
            for g, c in GROUPE_COULEURS.items()
            if g in groupes_affich
        ]
        legend_lines = [
            plt.Line2D([0], [0], color='#E89020', ls='--', lw=2,
                       label=f'Takt Time ({takt_yam}s)'),
            plt.Line2D([0], [0], color='#2FAE7A', ls=':', lw=2,
                       label=f'Cycle idéal ({CYCLE_IDEAL_YAM}s)'),
            mpatches.Patch(color='#D94030', label='Goulot (> Takt Time)'),
        ]
        ax_yam.legend(handles=legend_patches + legend_lines,
                      fontsize=8, loc='upper right',
                      ncol=min(len(legend_patches) + 3, 5))

        plt.tight_layout()
        st.pyplot(fig_yam)

        # Export graphique
        if st.button("📥 Exporter Yamazumi PNG", key="exp_yam_png"):
            fig_yam.savefig(f'NCAR_Yamazumi_S{sem_yam}.png', dpi=150, bbox_inches='tight')
            st.success(f"✅ 'NCAR_Yamazumi_S{sem_yam}.png' sauvegardé !")

    # ── KPIs & Analyse ─────────────────────────
    st.markdown('<p class="section-title">🔍 Analyse des goulots</p>', unsafe_allow_html=True)

    goulots_yam = []
    valeurs_non_nulles = []
    for poste in YAMAZUMI_POSTES_REF:
        nom = poste["station"]
        v63 = ct_values.get(nom, {}).get("ct63", 0) or 0
        v64 = ct_values.get(nom, {}).get("ct64", 0) or 0
        avg = (v63 + v64) / 2 if (v63 + v64) > 0 else 0
        ref_val = v63 if ref_filter == "Réf. 63" else v64 if ref_filter == "Réf. 64" else avg
        if ref_val > 0:
            valeurs_non_nulles.append(ref_val)
        if ref_val > takt_yam:
            goulots_yam.append({"station": nom, "groupe": poste["groupe"],
                                 "ct": ref_val, "ecart": ref_val - takt_yam})

    # KPI cards
    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    c_k1.markdown(
        f'<div class="kpi-card"><p class="kpi-label">Postes saisis</p>'
        f'<p class="kpi-val" style="color:#2B7FA8;">{len(valeurs_non_nulles)}</p>'
        f'<p class="kpi-sub">/ {len(YAMAZUMI_POSTES_REF)} postes</p></div>',
        unsafe_allow_html=True
    )
    c_k2.markdown(
        f'<div class="kpi-card"><p class="kpi-label">Goulots détectés</p>'
        f'<p class="kpi-val" style="color:#D94030;">{len(goulots_yam)}</p>'
        f'<p class="kpi-sub">CT > {takt_yam}s</p></div>',
        unsafe_allow_html=True
    )
    avg_ct = np.mean(valeurs_non_nulles) if valeurs_non_nulles else 0
    c_k3.markdown(
        f'<div class="kpi-card"><p class="kpi-label">CT Moyen ligne</p>'
        f'<p class="kpi-val" style="color:#1B6690;">{avg_ct:.1f}s</p>'
        f'<p class="kpi-sub">Takt Time : {takt_yam}s</p></div>',
        unsafe_allow_html=True
    )
    max_ct = max(valeurs_non_nulles) if valeurs_non_nulles else 0
    max_nom = noms_affich[vals_affich.index(max_ct)] if (valeurs_non_nulles and vals_affich) else "—"
    c_k4.markdown(
        f'<div class="kpi-card"><p class="kpi-label">CT Max (goulot principal)</p>'
        f'<p class="kpi-val" style="color:#E89020;">{max_ct:.0f}s</p>'
        f'<p class="kpi-sub">{max_nom}</p></div>',
        unsafe_allow_html=True
    )

    # Détail goulots
    if goulots_yam:
        st.markdown("<br>", unsafe_allow_html=True)
        for g in sorted(goulots_yam, key=lambda x: -x["ecart"]):
            col_g = GROUPE_COULEURS.get(g["groupe"], "#888")
            st.markdown(
                f'<div class="goulot-card" style="border-color:{col_g};">'
                f'<b style="color:{col_g};">⚠️ {g["station"]}</b> [{g["groupe"]}] — '
                f'CT : <b>{g["ct"]:.0f}s</b> | '
                f'Dépassement Takt Time : <b style="color:#E53935;">+{g["ecart"]:.1f}s</b>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.success("✅ Aucun goulot détecté — tous les postes sont sous le Takt Time.")

    # ── Comparaison multi-semaines ──────────────
    st.markdown('<p class="section-title">📅 Comparaison semaines (CT Moyen ligne)</p>', unsafe_allow_html=True)

    all_yam = charger_yamazumi()
    semaines_avec_data = sorted([k for k in all_yam.keys() if k.startswith("YAM_S")])

    if len(semaines_avec_data) >= 2:
        sems_comp, avg_comp, goulots_comp = [], [], []
        for sem_k in semaines_avec_data:
            sdata = all_yam[sem_k]
            vals_s = []
            nb_goulots_s = 0
            for poste in YAMAZUMI_POSTES_REF:
                nom = poste["station"]
                v63 = sdata.get(nom, {}).get("ct63", 0) or 0
                v64 = sdata.get(nom, {}).get("ct64", 0) or 0
                avg_s = (v63 + v64) / 2 if (v63 + v64) > 0 else 0
                if avg_s > 0:
                    vals_s.append(avg_s)
                if avg_s > takt_yam:
                    nb_goulots_s += 1
            if vals_s:
                sems_comp.append(sem_k.replace("YAM_", ""))
                avg_comp.append(np.mean(vals_s))
                goulots_comp.append(nb_goulots_s)

        if sems_comp:
            fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
            ax_comp.set_facecolor('#F3F8FB')
            x_c = np.arange(len(sems_comp))
            ax_comp.plot(x_c, avg_comp, 'o-', color='#2B7FA8', lw=2.5, ms=8, label='CT Moyen ligne (s)')
            ax_comp.axhline(y=takt_yam, color='#E89020', ls='--', lw=2, label=f'Takt Time ({takt_yam}s)')

            ax_c2 = ax_comp.twinx()
            ax_c2.bar(x_c, goulots_comp, alpha=0.25, color='#D94030', width=0.4, label='Nb goulots')
            ax_c2.set_ylabel('Nombre de goulots', color='#D94030')

            ax_comp.set_xticks(x_c)
            ax_comp.set_xticklabels(sems_comp, fontsize=9)
            ax_comp.set_ylabel('CT Moyen (s)')
            ax_comp.set_title('Évolution Yamazumi — Comparaison multi-semaines', fontweight='bold')
            h1, l1 = ax_comp.get_legend_handles_labels()
            h2, l2 = ax_c2.get_legend_handles_labels()
            ax_comp.legend(handles=h1 + h2, labels=l1 + l2, fontsize=9)
            ax_comp.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_comp)
    else:
        st.info("Enregistrez au moins 2 semaines pour afficher la comparaison.")

    # ── Export Excel ────────────────────────────
    st.markdown('<p class="section-title">📥 Export données Yamazumi</p>', unsafe_allow_html=True)
    if st.button("📥 Exporter Yamazumi Excel", key="exp_yam_xlsx"):
        rows_xlsx = []
        for poste in YAMAZUMI_POSTES_REF:
            nom = poste["station"]
            v63 = ct_values.get(nom, {}).get("ct63", 0) or 0
            v64 = ct_values.get(nom, {}).get("ct64", 0) or 0
            avg_e = (v63 + v64) / 2 if (v63 + v64) > 0 else 0
            rows_xlsx.append({
                "Semaine": f"S{sem_yam}",
                "Groupe":  poste["groupe"],
                "Poste":   nom,
                "CT Réf.63 (s)": v63,
                "CT Réf.64 (s)": v64,
                "CT Moyen (s)":  round(avg_e, 2),
                "Takt Time (s)": takt_yam,
                "Écart TT (s)":  round(avg_e - takt_yam, 2),
                "Statut":  "⚠️ Goulot" if avg_e > takt_yam else ("✅ OK" if avg_e > 0 else "—"),
            })
        pd.DataFrame(rows_xlsx).to_excel(f'NCAR_Yamazumi_S{sem_yam}.xlsx', index=False)
        st.success(f"✅ 'NCAR_Yamazumi_S{sem_yam}.xlsx' sauvegardé !")

# ══════════════════════════════════════════════
# ONGLET 10 — MODÈLE ML
# ══════════════════════════════════════════════
with tabs[9]:
    st.markdown('<p class="section-title">Comparaison des modèles</p>', unsafe_allow_html=True)
    rows_m = [{'Modèle':n, 'MAE CV-5':round(r['mae_cv'],3), 'R²':round(r['r2'],3),
               'Statut':'✅ Sélectionné' if n==nom_modele else ''}
              for n,r in resultats.items()]
    st.dataframe(pd.DataFrame(rows_m), use_container_width=True, hide_index=True)

    st.markdown('<p class="section-title">Importance des variables</p>', unsafe_allow_html=True)
    best_m = resultats[nom_modele]['model']
    if hasattr(best_m,'feature_importances_'):
        fi = pd.Series(best_m.feature_importances_, index=FEATURES).sort_values()
        labels_fr = {
            'demande_totale':'Demande totale','demande_ref63':'Demande Réf.63',
            'demande_ref64':'Demande Réf.64','stock':'Stock actuel',
            'cap_shift':'Capacité/shift','ratio_ref63':'Ratio Réf.63',
            'ratio_ref64':'Ratio Réf.64','shifts_base':'Shifts de base',
            'stock_insuffisant':'Stock insuffisant (<1089)','stock_suffisant':'Stock suffisant (≥1089)',
            'stock_ratio':'Ratio stock/seuil','stock_critique':'Stock critique (<100)',
            'correction_stock':'Correction stock','charge_index':'Indice de charge',
        }
        fig4,ax4 = plt.subplots(figsize=(9,6))
        cols_fi = ['#D94030' if 'stock' in f or 'cap' in f or 'ratio' in f else '#2B7FA8' for f in fi.index]
        brs = ax4.barh([labels_fr.get(f,f) for f in fi.index], fi.values, color=cols_fi, alpha=0.85, edgecolor='white')
        for bar,imp in zip(brs,fi.values):
            ax4.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f'{imp:.3f}', va='center', fontsize=9, fontweight='bold')
        ax4.set_xlabel('Importance'); ax4.set_xlim(0,0.65)
        ax4.set_title(f'Importance — {nom_modele}', fontweight='bold')
        ax4.legend(handles=[mpatches.Patch(color='#D94030',label='Capacité/stock/ratio'),
                              mpatches.Patch(color='#2B7FA8',label='Demande')], fontsize=9)
        plt.tight_layout(); st.pyplot(fig4)

    st.markdown('<p class="section-title">Paramètres du modèle</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.info(f"""
        **Constantes métier**
        - Stock sécurité : **{STOCK_SEC} u**
        - Réf.63 : **{CAP_H_REF63} câbles/h**
        - Réf.64 : **{CAP_H_REF64} câbles/h** (limité CCB)
        - Shift complet : **{SHIFT_HEURES}h** / **{OPE_COMPLET} opérateurs**
        - Demi-équipe : **4h** / **{OPE_DEMI} opérateurs**
        - Takt Time : **75.4s**
        """)
    with c2:
        st.success("""
        **Logique de correction stock**
        - Stock < 1089 → **+1 shift** (insuffisant)
        - Stock ≥ 1089 → **-1 shift** (suffisant)

        **Répartition Option A**
        - Phase 1 : 1 shift/jour Lu→Ve
        - Phase 2 : 2ème shift Lu→Ve
        - Phase 3 : Samedi si nécessaire
        """)