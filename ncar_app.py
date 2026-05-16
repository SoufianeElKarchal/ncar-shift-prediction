"""
=============================================================
  INTERFACE STREAMLIT — PRÉDICTION SHIFTS NCAR
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

# ─────────────────────────────────────────────
# GESTION DE L'AUTHENTIFICATION (FIXE ET CENTRALISÉ)
# ─────────────────────────────────────────────
if "autentifie" not in st.session_state:
    st.session_state["autentifie"] = False

def verifier_identifiants(username, password):
    USER_VALIDE = "admin"
    MDP_VALIDE = "yazaki2026"
    return username == USER_VALIDE and password == MDP_VALIDE

# Écran de connexion si l'utilisateur n'est pas connecté
if not st.session_state["autentifie"]:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* Masquer absolument tout le layout Streamlit en arrière-plan */
    [data-testid="stHeader"], [data-testid="stSidebar"], footer, .stDeployButton {
        display: none !important;
    }
    
    div[data-testid="stAppViewBlockContainer"] {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* POSITION FIXE INTERCEPTANT TOUT L'ÉCRAN : Évite le scroll en bas */
    .login-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #EFF6FA !important;
        z-index: 999999 !important;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: 'DM Sans', sans-serif !important;
        overflow: hidden;
    }

    /* Formes dégradées en arrière-plan fixe */
    .login-container::before {
        content: '';
        position: absolute;
        width: 450px; height: 450px;
        background: radial-gradient(circle, rgba(74,155,191,0.2) 0%, rgba(239,246,250,0) 70%);
        top: -100px; left: -100px;
        border-radius: 50%;
        filter: blur(40px);
    }
    
    .login-container::after {
        content: '';
        position: absolute;
        width: 550px; height: 550px;
        background: radial-gradient(circle, rgba(43,127,168,0.15) 0%, rgba(239,246,250,0) 70%);
        bottom: -150px; right: -100px;
        border-radius: 50%;
        filter: blur(40px);
    }

    /* Boîte blanche de connexion (Style épuré de ta maquette) */
    .login-card {
        position: relative;
        z-index: 10;
        background: #ffffff;
        padding: 50px 45px;
        border-radius: 24px;
        width: 100%;
        max-width: 440px;
        border: 1px solid #D9E8F0;
        box-shadow: 0 15px 35px rgba(27, 102, 144, 0.08), 
                    0 5px 15px rgba(27, 102, 144, 0.04);
        text-align: center;
        box-sizing: border-box;
    }

    .login-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 36px;
        font-weight: 700;
        color: #1a2e3b;
        margin-bottom: 6px;
    }
    
    .login-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        color: #7FA8BE;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 35px;
    }

    /* Nettoyage des bordures natives Streamlit */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    .login-card label {
        color: #1a2e3b !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-align: left !important;
        display: block !important;
        margin-bottom: 8px !important;
    }

    .login-card input {
        background-color: #ffffff !important;
        border: 1px solid #D9E8F0 !important;
        color: #1a2e3b !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-size: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.25s ease !important;
    }

    .login-card input:focus {
        border-color: #2B7FA8 !important;
        box-shadow: 0 0 0 4px rgba(43, 127, 168, 0.12) !important;
    }

    /* Bouton Connexion dégradé bleu original */
    .login-card button[data-testid="stFormSubmitButton"] {
        background: linear-gradient(135deg, #4A9BBF 0%, #2B7FA8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 15px !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100% !important;
        margin-top: 20px !important;
        cursor: pointer !important;
        box-shadow: 0 6px 20px rgba(43, 127, 168, 0.2) !important;
        transition: all 0.25s ease !important;
    }

    .login-card button[data-testid="stFormSubmitButton"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(43, 127, 168, 0.3) !important;
        filter: brightness(1.05);
    }

    .login-footer-text {
        margin-top: 30px;
        font-size: 12px;
        color: #7FA8BE;
    }
    
    .login-footer-text a {
        color: #2B7FA8;
        text-decoration: none;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Rendu HTML contrôlé de la carte de login au milieu
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Bienvenue</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">NCAR SHIFT SYSTEM — YAZAKI</div>', unsafe_allow_html=True)
    
    with st.form("yazaki_clean_login_form"):
        username_input = st.text_input("Identifiant ou Email", placeholder="Entrez votre identifiant...")
        password_input = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe...")
        bouton_connexion = st.form_submit_button("Se connecter")
        
        if bouton_connexion:
            if verifier_identifiants(username_input, password_input):
                st.session_state["autentifie"] = True
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")
                
    st.markdown('<div class="login-footer-text">Portail de production sécurisé. <a href="#">Aide technique</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# STYLE GLOBAL DE L'APPLICATION (APRES LOGIN)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #EFF6FA !important;
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
# FICHIER SUIVI JOURNALIER (persistance JSON)
# ─────────────────────────────────────────────
SUIVI_FILE = os.path.join(os.path.dirname(__file__), 'ncar_suivi.json')

def charger_suivi():
    if os.path.exists(SUIVI_FILE):
        with open(SUIVI_FILE) as f:
            return json.load(f)
    return {}

def sauvegarder_suivi(data):
    with open(SUIVI_FILE, 'w') as f:
        json.dump(data, f, indent=2)

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
    
    st.markdown("---")
    if st.button("🚪 Se déconnecter"):
        st.session_state["autentifie"] = False
        st.rerun()

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
                st.markdown(f'<div class="shift-matin"><b>{jour}</b><br><small style="color:#185FA5">Shift Matin</small><br><small>06h – 14h</small><br><b>8h</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="shift-matin"><b>{jour}</b><br><small style="color:#185FA5">Shift Matin</small><br><small>06h – 14h</small></div><div class="shift-apm"><small style="color:#1D9E75">Shift Après-midi</small><br><small>14h – 22h</small><br><b>16h total</b></div>', unsafe_allow_html=True)

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

# ─────────────────────────────────────────────
# ONGLET 3 — SUIVI JOURNALIER
# ─────────────────────────────────────────────
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

    st.dataframe(df_aff.style.map(
        lambda v: 'color:#C0392B;font-weight:600' if v=='⚠️ Écart'
                  else 'color:#1B5E20;font-weight:600' if v=='✅ Correct' else '',
        subset=['Statut ML']), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# ONGLET 6 — SEMAINES PASSÉES
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-title">Archive semaines passées</p>', unsafe_allow_html=True)
    st.info("Consultez les plannings archivés enregistrés dans le fichier ncar_suivi.json.")

# ══════════════════════════════════════════════
# ONGLET 7 — LIGNE NCAR
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-title">Équilibrage de la ligne NCAR — Yamazumi</p>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(12,6))
    ax.set_facecolor('#F3F8FB')
    ax.bar(['SPS1','SPS2','LAYOUT','FOAMING','TE','TV','CCB'], [62,68,70,71,78,80,112], color='#2B7FA8')
    ax.axhline(y=75.4,  color='#E89020', ls='--', lw=2.5, label='Takt Time (75.4s)')
    ax.legend(); st.pyplot(fig3)

# ══════════════════════════════════════════════
# ONGLET 8 — WIP (Encours de production)
# ══════════════════════════════════════════════
with tabs[7]:
    SEUIL_WIP = 8
    POSTES_WIP = [
        {'nom': 'SPS1'}, {'nom': 'SPS2'}, {'nom': 'LAYOUT'}, {'nom': 'FOAMING'},
        {'nom': 'AS1'}, {'nom': 'AS2'}, {'nom': 'AS3'}, {'nom': 'AS4'},
        {'nom': 'PRE'}, {'nom': 'CLP'}, {'nom': 'TE'}, {'nom': 'TV'},
        {'nom': 'CCB'}, {'nom': 'EOL'}, {'nom': 'VI'}, {'nom': 'PACK'}, {'nom': 'FW'}
    ]
    wip_key = f"WIP_S{semaine_num}"
    suivi_wip = charger_suivi()
    if wip_key not in suivi_wip:
        suivi_wip[wip_key] = {p['nom']: 0 for p in POSTES_WIP}

    cols_wip = st.columns(6)
    wip_vals = {}
    for i, poste in enumerate(POSTES_WIP):
        with cols_wip[i % 6]:
            val = st.number_input(poste['nom'], min_value=0, max_value=100, value=int(suivi_wip[wip_key].get(poste['nom'], 0)), key=f"wip_{poste['nom']}_{semaine_num}")
            wip_vals[poste['nom']] = val

    if st.button("💾 Sauvegarder WIP"):
        suivi_wip[wip_key] = wip_vals
        sauvegarder_suivi(suivi_wip)
        st.success("✅ Encours sauvegardés !")

# ══════════════════════════════════════════════
# ONGLET 9 — MODÈLE ML
# ══════════════════════════════════════════════
with tabs[8]:
    st.markdown('<p class="section-title">Paramètres du modèle ML</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{'Modèle': nom_modele, 'Status': 'Actif/Optimisé'}]), hide_index=True)