"""
=============================================================
  MODÈLE ML — PRÉDICTION DES SHIFTS OPTIMISÉS — LIGNE NCAR
  PFE : Optimisation du pilotage et de l'équilibrage NCAR

  Paramètres définitifs :
    - Réf. 63 : 49 câbles/heure
    - Réf. 64 : 33 câbles/heure (limité par CCB)
    - Stock sécurité : 1089 pièces
    - Correction : stock < 1089 → +1 | stock ≥ 1089 → -1
    - Demi-équipe : 32 opérateurs / 4h
    - Max shifts/jour : 2 | Répartition : Option A
=============================================================
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# CONSTANTES MÉTIER
# ─────────────────────────────────────────────
STOCK_SEC     = 1089
CAP_H_REF63   = 49
CAP_H_REF64   = 33
SHIFT_HEURES  = 8
DEMI_HEURES   = 4
MAX_SHIFTS_J  = 2
OPE_COMPLET   = 64
OPE_DEMI      = 32
JOURS_SEMAINE = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi']

# ─────────────────────────────────────────────
# DONNÉES RÉELLES (EDIS — Semaines 13 à 30)
# ─────────────────────────────────────────────
DONNEES_REELLES = pd.DataFrame({
    'semaine':        list(range(13, 31)),
    'demande_totale': [1030,1962,2076,2062,1683,2063,2150,1729,
                       2177,2447,1940,2355,2389,2366,1714,3066,2187,1309],
    'demande_ref63':  [202,871,952,946,853,970,885,768,
                       1017,1148,1135,1417,1442,1357,1025,1881,1304,776],
    'demande_ref64':  [828,1091,1124,1116,830,1093,1265,961,
                       1160,1299,805,938,947,1009,689,1185,883,533],
    'stock':          [0,0,0,122,189,121,34,143,
                       7,49,244,141,107,130,158,54,309,251],
    'my_vision':      [4,7,7,7,6,7,7,6,7,8,7,8,8,8,6,10,8,5],
})

# ─────────────────────────────────────────────
# FONCTIONS MÉTIER
# ─────────────────────────────────────────────
def cap_shift_reel(ref63, ref64):
    total = ref63 + ref64
    if total == 0:
        return round((CAP_H_REF63 + CAP_H_REF64) / 2 * SHIFT_HEURES)
    return round((ref63/total * CAP_H_REF63 + ref64/total * CAP_H_REF64) * SHIFT_HEURES)

def cap_demie_equipe(ref63, ref64):
    total = ref63 + ref64
    if total == 0:
        return round((CAP_H_REF63 + CAP_H_REF64) / 2 * DEMI_HEURES)
    return round((ref63/total * CAP_H_REF63 + ref64/total * CAP_H_REF64) * DEMI_HEURES)

def calculer_shifts_optimises(demande, stock, ref63, ref64):
    cap  = cap_shift_reel(ref63, ref64)
    base = int(np.ceil(demande / cap))
    corr = +1 if stock < STOCK_SEC else -1
    return max(1, base + corr), cap

def repartir_shifts(total_shifts):
    jours   = [0] * 6
    restant = int(total_shifts)
    for i in range(5):
        if restant > 0: jours[i] = 1; restant -= 1
    for i in range(5):
        if restant > 0 and jours[i] < MAX_SHIFTS_J: jours[i] = 2; restant -= 1
    if restant > 0: jours[5] = 1; restant -= 1
    if restant > 0: jours[5] = 2
    return jours

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def make_features(df):
    df = df.copy()
    df['cap_shift']         = df.apply(lambda r: cap_shift_reel(r['demande_ref63'], r['demande_ref64']), axis=1)
    df['ratio_ref63']       = df['demande_ref63'] / df['demande_totale'].replace(0, 1)
    df['ratio_ref64']       = df['demande_ref64'] / df['demande_totale'].replace(0, 1)
    df['shifts_base']       = df['demande_totale'] / df['cap_shift']
    df['stock_insuffisant'] = (df['stock'] < STOCK_SEC).astype(int)
    df['stock_suffisant']   = (df['stock'] >= STOCK_SEC).astype(int)
    df['stock_ratio']       = df['stock'] / STOCK_SEC
    df['stock_critique']    = (df['stock'] < 100).astype(int)
    df['correction_stock']  = df['stock_insuffisant'] * 1 + df['stock_suffisant'] * (-1)
    df['charge_index']      = df['demande_totale'] / (df['cap_shift'] * 5)
    return df

FEATURES = [
    'demande_totale','demande_ref63','demande_ref64',
    'stock','cap_shift','ratio_ref63','ratio_ref64',
    'shifts_base','stock_insuffisant','stock_suffisant',
    'stock_ratio','stock_critique','correction_stock','charge_index'
]

# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
def generer_donnees_augmentees(n=300, seed=42):
    np.random.seed(seed)
    dem_aug   = np.random.normal(2039, 450, n).clip(800, 3500)
    stk_aug   = np.random.uniform(0, 600, n)
    r63_ratio = np.random.uniform(0.3, 0.7, n)
    ref63_aug = dem_aug * r63_ratio
    ref64_aug = dem_aug * (1 - r63_ratio)

    aug = pd.DataFrame({'demande_totale': dem_aug, 'demande_ref63': ref63_aug,
                        'demande_ref64': ref64_aug, 'stock': stk_aug})
    aug = make_features(aug)
    aug['shifts_opt'] = [calculer_shifts_optimises(d, s, r3, r4)[0]
                         for d, s, r3, r4 in zip(dem_aug, stk_aug, ref63_aug, ref64_aug)]
    noise = np.random.choice([-1,0,0,0,1], size=n)
    aug['shifts_opt'] = (aug['shifts_opt'] + noise).clip(3, 14)
    return aug

# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────
def entrainer_modele(verbose=True):
    df_real = DONNEES_REELLES.copy()
    df_real['shifts_opt'] = [
        calculer_shifts_optimises(d, s, r3, r4)[0]
        for d, s, r3, r4 in zip(df_real['demande_totale'], df_real['stock'],
                                  df_real['demande_ref63'], df_real['demande_ref64'])
    ]
    df_real = make_features(df_real)
    df_aug  = generer_donnees_augmentees(n=300)
    df      = pd.concat([df_real[FEATURES+['shifts_opt']],
                         df_aug[FEATURES+['shifts_opt']]], ignore_index=True)
    X, y = df[FEATURES], df['shifts_opt']

    modeles = {
        'Random Forest':     RandomForestRegressor(n_estimators=300, max_depth=7, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'Linear Regression': LinearRegression(),
    }

    if verbose:
        print("="*58)
        print("  MODÈLE ML NCAR — Paramètres définitifs")
        print(f"  Stock sec : {STOCK_SEC}u | Réf63 : {CAP_H_REF63}/h | Réf64 : {CAP_H_REF64}/h")
        print("="*58)

    best_name, best_mae, best_model = None, 999, None
    resultats = {}
    for name, model in modeles.items():
        cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        resultats[name] = {'mae_cv': cv_mae, 'r2': r2, 'model': model}
        if verbose: print(f"\n  {name}\n    MAE CV-5 : {cv_mae:.3f} | R² : {r2:.3f}")
        if cv_mae < best_mae: best_mae, best_name, best_model = cv_mae, name, model

    if verbose:
        print(f"\n{'='*58}\n  MEILLEUR : {best_name}  (MAE = {best_mae:.3f})\n{'='*58}\n")
        X_real = df_real[FEATURES]; y_real = df_real['shifts_opt']
        y_pred = np.round(best_model.predict(X_real)).astype(int)
        exact  = (y_pred == y_real.values).sum()
        print("── Validation 18 semaines ──")
        print(f"{'Sem.':<6}{'Dem.':<7}{'Stock':<7}{'Cap/sh':<8}{'MY_VIS':<9}{'OPT':<6}{'PRÉDIT':<8}{'Statut'}")
        print("-"*58)
        for i in range(len(df_real)):
            flag = "✅" if y_pred[i]==y_real.values[i] else f"⚠️({y_pred[i]-y_real.values[i]:+})"
            print(f"S{int(df_real['semaine'].values[i]):<5}"
                  f"{int(df_real['demande_totale'].values[i]):<7}"
                  f"{int(df_real['stock'].values[i]):<7}"
                  f"{int(df_real['cap_shift'].values[i]):<8}"
                  f"{int(df_real['my_vision'].values[i]):<9}"
                  f"{int(y_real.values[i]):<6}{int(y_pred[i]):<8}{flag}")
        print(f"\n  Exactes : {exact}/18 ({exact/18*100:.0f}%)  |  MAE : {mean_absolute_error(y_real,y_pred):.2f}")

    return best_model, best_name, resultats

# ─────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────
def predire_shifts(modele, demande, stock, ref63, ref64):
    cap   = cap_shift_reel(ref63, ref64)
    cap_d = cap_demie_equipe(ref63, ref64)
    opt,_ = calculer_shifts_optimises(demande, stock, ref63, ref64)
    repart = repartir_shifts(opt)
    base  = int(np.ceil(demande / cap))

    row = pd.DataFrame([{
        'demande_totale':   demande, 'demande_ref63': ref63, 'demande_ref64': ref64,
        'stock':            stock,   'cap_shift':     cap,
        'ratio_ref63':      ref63/max(ref63+ref64,1), 'ratio_ref64': ref64/max(ref63+ref64,1),
        'shifts_base':      demande/cap,
        'stock_insuffisant':int(stock < STOCK_SEC), 'stock_suffisant': int(stock >= STOCK_SEC),
        'stock_ratio':      stock/STOCK_SEC, 'stock_critique': int(stock < 100),
        'correction_stock': 1 if stock < STOCK_SEC else -1,
        'charge_index':     demande/(cap*5),
    }])
    pred = int(np.round(modele.predict(row[FEATURES])[0]))

    return {
        'shifts_predits':    pred,
        'shifts_my_vision':  base,
        'shifts_optimises':  opt,
        'correction':        opt - base,
        'cap_shift':         cap,
        'cap_demie':         cap_d,
        'repartition':       dict(zip(JOURS_SEMAINE, repart)),
        'jours_actifs':      sum(1 for x in repart if x > 0),
        'heures_production': opt * SHIFT_HEURES,
        'stock_statut':      'insuffisant' if stock < STOCK_SEC else 'suffisant',
    }

def sauvegarder_modele(modele, chemin='ncar_modele.pkl'):
    with open(chemin, 'wb') as f: pickle.dump(modele, f)
    print(f"✅ Modèle sauvegardé : {chemin}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    modele, nom, resultats = entrainer_modele(verbose=True)
    sauvegarder_modele(modele)
    print("\n── Test prédiction (S31 fictive) ──")
    res = predire_shifts(modele, 2200, 80, 1100, 1100)
    print(f"  Demande : 2200u | Stock : 80u | Cap/shift : {res['cap_shift']}u")
    print(f"  MY VISION : {res['shifts_my_vision']} | OPTIMISÉ : {res['shifts_optimises']} (+1)")
    for jour, s in res['repartition'].items():
        txt = '2 shifts (06h–22h)' if s==2 else '1 shift (06h–14h)' if s==1 else 'Repos'
        print(f"    {jour:<12} : {txt}")
