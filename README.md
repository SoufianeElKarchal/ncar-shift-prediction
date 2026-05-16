# 🏭 Prédiction des Shifts et Pilotage Logistique — Ligne NCAR (YAZAKI)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

Ce dépôt héberge l'application web industrielle interactive développée avec **Streamlit** et propulsée par du **Machine Learning** dans le cadre d'un Projet de Fin d'Études (PFE) axé sur la digitalisation et l'optimisation des flux au sein du groupe **YAZAKI**.

L'application permet d'optimiser le pilotage de la ligne de production **NCAR** en prédisant le nombre exact de shifts (équipes) requis chaque semaine en fonction de la demande client, tout en lissant la charge de travail, en réduisant les coûts de main-d'œuvre et en sécurisant les niveaux de stock.

---

## 🚀 Modules & Fonctionnalités de l'Application

* **📅 Calendrier de Production** : Génération et répartition automatique des plannings (Matin, Après-midi, Repos) sur la semaine pour les opérateurs.
* **📊 Suivi Journalier** : Tableau de bord comparatif mesurant en temps réel les écarts entre les objectifs logistiques fixés et le réalisé terrain.
* **💡 Simulation What-If** : Outil d'aide à la décision permettant de faire varier la demande client ou l'état des stocks pour observer l'impact direct sur les besoins en ressources humaines.
* **📉 Yamazumi Chart & Équilibrage** : Visualisation dynamique des temps de cycle par poste pour identifier et éliminer les goulots d'étranglement (Temps Manuel, Temps Technologique, contraintes machine CCB).
* **🗺️ Cartographie Dynamic WIP** : Représentation 2D interactive des encours (*Work In Progress*) sur la ligne NCAR avec déclenchement d'alertes visuelles de saturation.

---

## ⚙️ Spécifications Métier & Constantes Usine

L'ensemble de la logique algorithmique respecte scrupuleusement les contraintes opérationnelles de la ligne NCAR :

* **Stock de Sécurité ($S_s$)** : $1089$ pièces (seuil critique réglementaire de déclenchement des alertes).
* **Cadence Référence 63** : $49$ câbles / heure.
* **Cadence Référence 64** : $33$ câbles / heure *(cadence bridée par la station critique CCB)*.
* **Format d'Équipe Standard** : Équipe complète ($64$ opérateurs / shift de $8\text{h}$) ou demi-équipe ($32$ opérateurs / shift de $4\text{h}$).
* **Règle de Correction Automatique** : Si $\text{Stock} < 1089 \rightarrow +1 \text{ Shift}$, si $\text{Stock} \ge 1089 \rightarrow -1 \text{ Shift}$.

---

## 🛠️ Installation et Lancement Local (Sous Windows)

Suivez ces étapes depuis votre terminal (**PowerShell** ou **Invite de commandes**) pour exécuter le projet localement :

### 1. Clonage du Dépôt

```bash
git clone [https://github.com/SoufianeElKarchal/ncar-shift-prediction.git](https://github.com/SoufianeElKarchal/ncar-shift-prediction.git)
cd ncar-shift-prediction
```
