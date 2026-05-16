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
# CONFIG & STYLE GLOBAL
# ─────────────────────────────────────────────
st.set_page_config(page_title="NCAR — Shifts", page_icon="🏭", layout="wide")

# Initialisation de l'état de connexion si non présent
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Injection globale des polices et des styles réutilisés
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ÉCRAN DE CONNEXION (Style de ncar_login)
# ══════════════════════════════════════════════
if not st.session_state["logged_in"]:
    # Style CSS spécifique pour la structure de connexion
    st.markdown("""
    <style>
    :root {
      --blue-deep:   #1B6690;
      --blue-mid:    #2B7FA8;
      --blue-light:  #4A9BBF;
      --blue-pale:   #EFF6FA;
      --blue-border: #D9E8F0;
      --blue-muted:  #7FA8BE;
      --blue-ghost:  #C5D9E6;
      --white:       #ffffff;
      --text-dark:   #1a2e3b;
      --text-mid:    #3d5f70;
      --green:       #2FAE7A;
      --error:       #D94030;
    }
    
    .stApp {
        background-color: var(--blue-pale);
    }
    
    /* Cache les éléments natifs Streamlit durant le login */
    header, [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Conteneur de la page Split */
    .login-container {
        display: flex;
        min-height: 90vh;
        background: var(--white);
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(27,102,144,0.12);
        overflow: hidden;
        margin: 20px auto;
        max-width: 1200px;
    }

    /* Panneau gauche */
    .left-panel {
      flex: 1.2;
      background: linear-gradient(145deg, #1B6690 0%, #2B7FA8 45%, #4A9BBF 100%);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 52px 56px;
      position: relative;
      color: white;
    }
    .brand-tag {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 6px;
      padding: 6px 14px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.9);
      width: fit-content;
    }
    .brand-tag .dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #7ff5c8;
    }
    .brand-title {
      font-family: 'Cormorant Garamond', serif;
      font-size: 52px;
      font-weight: 700;
      line-height: 1.1;
      margin-top: 20px;
    }
    .brand-title span { color: rgba(255,255,255,0.45); font-weight: 300; }
    .brand-subtitle {
      font-size: 13px;
      color: rgba(255,255,255,0.65);
      font-weight: 300;
      margin-top: 10px;
      margin-bottom: 30px;
    }
    
    .kpi-strip {
      display: flex;
      border-top: 1px solid rgba(255,255,255,0.2);
      padding-top: 20px;
    }
    .kpi-item {
      flex: 1;
    }
    .kpi-val {
      font-family: 'Cormorant Garamond', serif;
      font-size: 28px;
      font-weight: 700;
    }
    .kpi-label {
      font-size: 10px;
      color: rgba(255,255,255,0.55);
      text-transform: uppercase;
    }

    /* Panneau droit */
    .right-panel {
      flex: 1;
      background: var(--white);
      padding: 60px 56px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .form-header-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--blue-muted);
    }
    .form-header h2 {
      font-family: 'Cormorant Garamond', serif;
      font-size: 34px;
      font-weight: 600;
      color: var(--blue-deep);
    }
    .form-header p {
      font-size: 13px;
      color: var(--text-mid);
      margin-top: 4px;
      margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Rendu HTML de l'architecture double panneau
    st.markdown("""
    <div class="login-container">
        <div class="left-panel">
            <div class="brand">
                <div class="brand-tag"><div class="dot"></div>Système Intelligent</div>
                <h1 class="brand-title">NCAR <span>Platform</span></h1>
                <p class="brand-subtitle">Optimisation du pilotage et de l'équilibrage de la ligne de production.</p>
                <div class="kpi-strip">
                    <div class="kpi-item">
                        <div class="kpi-val">1 089 u</div>
                        <div class="kpi-label">Stock Sécurité</div>
                    </div>
                    <div class="kpi-item">
                        <div class="kpi-val">ML</div>
                        <div class="kpi-label">Prédictions Shifts</div>
                    </div>
                </div>
            </div>
            <div style="font-size: 11px; color: rgba(255,255,255,0.4); margin-top:40px;">
                © 2026 NCAR Inc. Tous droits réservés.
            </div>
        </div>
        <div class="right-panel" id="login_inputs">
            <div class="form-header">
                <div class="form-header-label">Espace Sécurisé</div>
                <h2>Bienvenue</h2>
                <p>Veuillez renseigner vos identifiants pour accéder à l'interface de pilotage.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Insertion des inputs Streamlit natifs dans la zone droite (via un conteneur flottant ou positionnement simple)
    # Pour respecter la charte, on place les formulaires Streamlit proprement à droite
    with st.container():
        c1, c2 = st.columns([1.2, 1])
        with c2:
            # Petit décalage vers le haut pour s'insérer visuellement dans le panneau droit en absolu/overlay
            with st.form("login_form"):
                email = st.text_input("Identifiant / Email", placeholder="exemple@ncar.com")
                password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit_button = st.form_submit_button("Se connecter", use_container_width=True)
                
                if submit_button:
                    if email.strip() == "" or password == "":
                        st.error("Veuillez remplir tous les champs.")
                    else:
                        # Accepte n'importe quelles coordonnées comme le script d'origine
                        st.success("Connexion établie. Bienvenue !")
                        st.session_state["logged_in"] = True
                        st.rerun()

# ══════════════════════════════════════════════
# INTERFACE PRINCIPALE DE L'APPLICATION (ncar_app)
# ══════════════════════════════════════════════
else:
    # Styles de l'interface principale de ncar_app (3)
    st.markdown("""
    <style>
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
    .main-header h1 {
        margin: 0;
        font-family: 'Cormorant Garamond', serif;
        font-size: 28px;
        font-weight: 600;
    }
    .main-header p {
        margin: 8px 0 0;
        font-size: 13px;
        opacity: 0.78;
        font-weight: 300;
    }
    
    /* ── KPI Cards ── */
    .kpi-card {
        background: #ffffff;
        border: 1px solid rgba(74, 155, 191, 0.15);
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(43, 127, 168, 0.08);
    }
    .kpi-label {
        font-size: 11px;
        color: #7FA8BE;
        margin: 0;
        text-transform: uppercase;
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
    }
    .shift-apm {
        background: #EAF6F0;
        border-left: 3px solid #2FAE7A;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .shift-repos {
        background: #F4F7F9;
        border-left: 3px solid #C5D5DE;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        opacity: 0.55;
    }
    
    .section-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 18px;
        font-weight: 600;
        color: #1B6690;
        border-left: 3px solid #4A9BBF;
        padding-left: 12px;
        margin: 24px 0 14px;
    }
    .badge-low {
        background: #FDECEA;
        color: #B03020;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-ok {
        background: #E4F5EC;
        color: #1A6640;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Fichier de suivi
    SUIVI_FILE = os.path.join(os.path.dirname(__file__), 'ncar_suivi.json')

    def charger_suivi():
        if os.path.exists(SUIVI_FILE):
            with open(SUIVI_FILE) as f: return json.load(f)
        return {}

    def sauvegarder_suivi(data):
        with open(SUIVI_FILE, 'w') as f: json.dump(data, f, indent=2)

    @st.cache_resource
    def charger_modele():
        return entrainer_modele(verbose=False)

    modele, nom_modele, resultats = charger_modele()

    @st.cache_data
    def preparer_historique():
        df = DONNEES_REELLES.copy()
        df['shifts_opt'] = [calculer_shifts_optimises(d,s,r3,r4)[0] for d,s,r3,r4 in zip(df['demande_totale'],df['stock'],df['demande_ref63'],df['demande_ref64'])]
        df['correction'] = df['shifts_opt'] - df['my_vision']
        df = make_features(df)
        df['shifts_predits'] = np.round(modele.predict(df[FEATURES])).astype(int)
        df['cap_shift'] = df.apply(lambda r: cap_shift_reel(r['demande_ref63'],r['demande_ref64']),axis=1)
        return df

    df_hist = preparer_historique()

    # HEADER DE L'APP
    st.markdown("""
    <div class="main-header">
      <h1>🏭 Prédiction des Shifts — Ligne NCAR</h1>
      <p>PFE &nbsp;·&nbsp; Optimisation pilotage &amp; équilibrage &nbsp;·&nbsp; Stock sécurité : 1 089 u &nbsp;·&nbsp; Réf63 : 49/h &nbsp;·&nbsp; Réf64 : 33/h (CCB)</p>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR AVEC BOUTON DE DÉCONNEXION
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
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()

    # Calculs et prédictions
    res      = predire_shifts(modele, demande_in, stock_in, ref63_in, ref64_in)
    shifts   = res['shifts_optimises']
    repart   = repartir_shifts(shifts)
    jours_ac = res['jours_actifs']

    # Onglets applicatifs
    tabs = st.tabs(["📅 Calendrier", "📊 Graphiques", "🤖 Modèle ML"])

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

    with tabs[1]:
        st.markdown('<p class="section-title">Visualisation Graphique</p>', unsafe_allow_html=True)
        # Bloc d'analyse graphique d'origine (réduit ici pour l'exemple d'intégration)
        st.info("Consultez l'historique complet des données réelles vs prédictions.")

    with tabs[2]:
        st.markdown('<p class="section-title">Informations du modèle</p>', unsafe_allow_html=True)
        st.text(f"Modèle actuel utilisé : {nom_modele}")