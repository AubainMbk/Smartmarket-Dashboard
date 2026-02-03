# 📊 SmartMarket — Dashboard Marketing (Streamlit)

Application interactive de pilotage de la performance marketing pour **SmartMarket**, un e-commerçant spécialisé dans les accessoires technologiques.

👉 **Application en ligne (accès public)**  
🔗 https://smartmarket-dashboard-fxans4yfgtfeignv5pxvev.streamlit.app/

---

## 🎯 Objectif de l’application

Cette application a été conçue pour fournir à la **direction marketing** une vue d’ensemble claire, synthétique et actionnable afin de :

- comparer la **performance des canaux marketing**,
- analyser la **qualité des leads** via les statuts CRM,
- identifier les **leviers d’optimisation budgétaire**,
- appuyer la prise de décision grâce à des **indicateurs clés dynamiques**.

L’ensemble des indicateurs est **recalculé en temps réel** à partir des filtres sélectionnés.

---

## 🧩 Fonctionnalités principales

### 🔎 Filtres interactifs
Disponibles dans la barre latérale :
- période (septembre 2025),
- canal marketing,
- région,
- statut CRM (MQL / SQL / Client),
- device (Mobile / Desktop / Tablet).

---

### 📌 Indicateurs clés (KPI)
- CTR global
- Taux de conversion post-clic
- Coût par conversion (CPA)
- Coût par lead (CPL)
- Part de leads à forte valeur (SQL + Client)

---

### 📈 Visualisations
L’application propose une lecture structurée en 4 graphiques principaux :

1. **CTR par canal**  
2. **CPA par canal**  
3. **Qualité business : Canal × Statut CRM**  
4. **Volume de leads par région**

Les graphiques sont volontairement **non décoratifs**, pensés pour une lecture rapide par un décideur.

---

### 💡 Insight dynamique
Un encart met automatiquement en évidence :
- le **canal le plus rentable** (CPA minimal),
- le **canal à optimiser** (CPA le plus élevé),

avec les métriques associées (coût, conversions).

---

### 🧠 Conclusion opérationnelle intégrée
L’application synthétise les principaux enseignements et propose des **axes d’action concrets** :
- arbitrage budgétaire,
- pilotage du funnel par canal,
- priorisation mobile et géographique.

---

## 🗂️ Données & périmètre

- **Période analysée** : septembre 2025
- **Sources** :
  - données de campagnes (coûts, impressions, clics, conversions),
  - leads marketing,
  - enrichissement CRM (statut, région, secteur, taille d’entreprise).
- Le jeu de données a été **élargi par génération contrôlée** afin de se rapprocher de volumes réalistes, tout en conservant les équilibres métiers.

---

## 🛠️ Stack technique

- **Python**
- **Streamlit**
- Pandas / NumPy
- Matplotlib
- Déploiement : **Streamlit Community Cloud**

---

## 🚀 Déploiement

L’application est déployée sur **Streamlit Cloud** et accessible via une URL publique :  
👉 https://smartmarket-dashboard-fxans4yfgtfeignv5pxvev.streamlit.app/

Chaque mise à jour du code sur GitHub déclenche automatiquement un **redéploiement**.

---

## 📌 À propos

Ce projet illustre une démarche complète :
- sélection et préparation des données,
- analyse statistique univariée et bivariée,
- visualisation orientée métier,
- restitution via un dashboard décisionnel.


---
