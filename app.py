import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------
# Page config (best practice)
# -----------------------
st.set_page_config(
    page_title="Dashboard Marketing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# Style léger pour compacter (best practice)
# -----------------------
st.markdown(
    """
    <style>
      /* Réduit l'espace vertical global */
      .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
      /* Rend les métriques plus compactes */
      [data-testid="stMetricValue"] { font-size: 1.35rem; }
      [data-testid="stMetricLabel"] { font-size: 0.95rem; }
      /* Réduit l'espace entre éléments */
      div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Chargement & préparation
# -----------------------
@st.cache_data
def load_data():
    leads = pd.read_csv("data/leads_smartmarket_extended.csv", parse_dates=["date"])
    crm = pd.read_excel("data/crm_smartmarket_extended.xlsx")
    with open("data/campaign_smartmarket.json", "r", encoding="utf-8") as f:
        campaigns = pd.DataFrame(json.load(f))


    def norm_cols(df):
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
        return df

    leads = norm_cols(leads)
    crm = norm_cols(crm)
    campaigns = norm_cols(campaigns)

    # Périmètre Septembre 2025
    leads = leads[(leads["date"] >= "2025-09-01") & (leads["date"] <= "2025-09-30")].copy()

    # Nettoyage des catégories (supprime espaces parasites)
    for col in ["channel", "device"]:
        if col in leads.columns:
            leads[col] = leads[col].astype(str).str.strip()

    for col in ["region", "status", "company_size", "sector", "city"]:
        if col in crm.columns:
            crm[col] = crm[col].astype(str).str.strip()

    if "channel" in campaigns.columns:
        campaigns["channel"] = campaigns["channel"].astype(str).str.strip()


    # Jointure CRM (table "lead enrichi")
    leads_enriched = leads.merge(crm, on="lead_id", how="left", validate="one_to_one")

    return leads_enriched, campaigns

leads_enriched, campaigns = load_data()

# -----------------------
# Sidebar : filtres (best practice)
# -----------------------
st.sidebar.header("Filtres")

# Plage de dates (dans le périmètre déjà filtré Sept 2025)
min_date = leads_enriched["date"].min()
max_date = leads_enriched["date"].max()
date_range = st.sidebar.date_input(
    "Période",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

# Multiselects
all_channels = sorted(leads_enriched["channel"].dropna().unique().tolist())
all_regions = sorted(leads_enriched["region"].dropna().unique().tolist()) if "region" in leads_enriched.columns else []
all_status = sorted(leads_enriched["status"].dropna().unique().tolist()) if "status" in leads_enriched.columns else []
all_devices = sorted(leads_enriched["device"].dropna().unique().tolist()) if "device" in leads_enriched.columns else []

sel_channels = st.sidebar.multiselect("Canaux", options=all_channels, default=all_channels)
sel_regions = st.sidebar.multiselect("Régions", options=all_regions, default=all_regions) if all_regions else []
sel_status = st.sidebar.multiselect("Statut CRM", options=all_status, default=all_status) if all_status else []
sel_devices = st.sidebar.multiselect("Device", options=all_devices, default=all_devices) if all_devices else []

# Application des filtres aux leads
df = leads_enriched.copy()

# Date filter
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d = pd.to_datetime(date_range[0])
    end_d = pd.to_datetime(date_range[1])
    df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]

# Cat filters
if sel_channels:
    df = df[df["channel"].isin(sel_channels)]
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_status:
    df = df[df["status"].isin(sel_status)]
if sel_devices:
    df = df[df["device"].isin(sel_devices)]

# Pour les campagnes : on filtre sur les canaux sélectionnés (cohérence dashboard)
camp = campaigns.copy()
if sel_channels:
    camp = camp[camp["channel"].isin(sel_channels)]

# -----------------------
# Fonctions KPI
# -----------------------
def safe_div(a, b):
    return (a / b) if b not in (0, None) and not pd.isna(b) else np.nan

# Agrégats campagne
total_impr = camp["impressions"].sum() if len(camp) else 0
total_clicks = camp["clicks"].sum() if len(camp) else 0
total_conv = camp["conversions"].sum() if len(camp) else 0
total_cost = camp["cost"].sum() if len(camp) else 0

# Leads
total_leads = df["lead_id"].nunique()

ctr_global = safe_div(total_clicks, total_impr)
cvr_postclick_global = safe_div(total_conv, total_clicks)
cpa_global = safe_div(total_cost, total_conv)
cpl_global = safe_div(total_cost, total_leads)

# Qualité business
sql_count = (df["status"] == "SQL").sum() if "status" in df.columns else 0
client_count = (df["status"] == "Client").sum() if "status" in df.columns else 0
high_value_rate = safe_div((sql_count + client_count), total_leads)

# KPI par canal (campagnes)
kpi = pd.DataFrame(columns=["channel", "ctr", "cpa"])
if len(camp):
    kpi = camp.copy()
    kpi["ctr"] = np.where(kpi["impressions"] > 0, kpi["clicks"] / kpi["impressions"], np.nan)
    kpi["cpa"] = np.where(kpi["conversions"] > 0, kpi["cost"] / kpi["conversions"], np.nan)

# Crosstab canal x statut (leads)
ct_channel_status = pd.crosstab(df["channel"], df["status"]) if ("status" in df.columns and len(df)) else pd.DataFrame()

# -----------------------
# Header
# -----------------------
st.title("Dashboard Marketing (Septembre 2025)")
st.caption("Vue synthétique : performance des canaux, qualité CRM, focus géographique. Les filtres à gauche recalculent tous les indicateurs.")

# -----------------------
# Bandeau KPI (compact)
# -----------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("CTR global", f"{ctr_global*100:.2f}%" if not pd.isna(ctr_global) else "—")
k2.metric("Conv. post-clic", f"{cvr_postclick_global*100:.2f}%" if not pd.isna(cvr_postclick_global) else "—")
k3.metric("CPA global", f"{cpa_global:.2f} €" if not pd.isna(cpa_global) else "—")
k4.metric("CPL global", f"{cpl_global:.2f} €" if not pd.isna(cpl_global) else "—")
k5.metric("% SQL + Client", f"{high_value_rate*100:.1f}%" if not pd.isna(high_value_rate) else "—")

st.divider()

# -----------------------
# Layout principal : 2 lignes de visuels (lisible d’un coup)
# -----------------------
row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)

# Graph 1: CTR par canal
with row1_left:
    st.subheader("CTR par canal")
    if len(kpi):
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        d = kpi.sort_values("ctr", ascending=False)
        ax.bar(d["channel"], d["ctr"])
        ax.set_title("CTR (clicks / impressions)", fontsize=11)
        ax.set_xlabel("Canal")
        ax.set_ylabel("CTR")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Pas de données campagnes pour les filtres actuels.")

# Graph 2: CPA par canal
with row1_right:
    st.subheader("CPA par canal")
    if len(kpi):
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        d = kpi.sort_values("cpa", ascending=True)
        ax.bar(d["channel"], d["cpa"])
        ax.set_title("CPA (€) (cost / conversions)", fontsize=11)
        ax.set_xlabel("Canal")
        ax.set_ylabel("CPA (€)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Pas de données campagnes pour les filtres actuels.")

# Graph 3: Canal x Statut
with row2_left:
    st.subheader("Qualité business : Canal × Statut")
    if not ct_channel_status.empty:
        # Hauteur/largeur uniformes
        fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=140)

        ct = ct_channel_status.copy()

        # Ordre des colonnes stable (statuts)
        status_order = [c for c in ["MQL", "SQL", "Client"] if c in ct.columns]
        if status_order:
            ct = ct[status_order]

        # Ordre des lignes stable (canaux)
        channel_order = [c for c in ["Emailing", "Facebook Ads", "Instagram Ads", "LinkedIn"] if c in ct.index]
        if channel_order:
            ct = ct.reindex(channel_order)

        ct.plot(kind="bar", stacked=True, ax=ax, width=0.7)

        ax.set_title("Répartition des statuts par canal", fontsize=11)
        ax.set_xlabel("Canal")
        ax.set_ylabel("Nombre de leads")
        ax.tick_params(axis="x", rotation=20)

        # Légende en bas -> évite le décalage à droite
        ax.legend(
            title="Statut",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(3, len(ct.columns)),
            frameon=False
        )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Statut CRM indisponible ou aucun lead après filtrage.")


# Graph 4: Leads par région
with row2_right:
    st.subheader("Leads par région")
    if "region" in df.columns and len(df):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        region_counts = df["region"].value_counts()
        ax.bar(region_counts.index.astype(str), region_counts.values)
        ax.set_title("Volume de leads par région", fontsize=11)
        ax.set_xlabel("Région")
        ax.set_ylabel("Nombre de leads")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Région indisponible ou aucun lead après filtrage.")

st.divider()

# -----------------------
# Insight dynamique (compact) : meilleur canal vs canal à optimiser (CPA)
# -----------------------
st.subheader("Insight dynamique (CPA)")

if len(kpi):
    # On exclut les canaux sans conversions (CPA = NaN)
    kpi_valid = kpi.dropna(subset=["cpa"]).copy()

    if len(kpi_valid):
        best = kpi_valid.sort_values("cpa", ascending=True).iloc[0]
        worst = kpi_valid.sort_values("cpa", ascending=False).iloc[0]

        best_channel = best["channel"]
        best_cpa = float(best["cpa"])
        best_conv = int(best["conversions"])
        best_cost = float(best["cost"])

        worst_channel = worst["channel"]
        worst_cpa = float(worst["cpa"])
        worst_conv = int(worst["conversions"])
        worst_cost = float(worst["cost"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.success(
                f"**Meilleur canal (CPA minimal)** : **{best_channel}**\n\n"
                f"- CPA : **{best_cpa:.2f} €**\n"
                f"- Conversions : **{best_conv}**\n"
                f"- Dépenses : **{best_cost:.0f} €**"
            )

        with col_b:
            st.warning(
                f"**Canal à optimiser (CPA élevé)** : **{worst_channel}**\n\n"
                f"- CPA : **{worst_cpa:.2f} €**\n"
                f"- Conversions : **{worst_conv}**\n"
                f"- Dépenses : **{worst_cost:.0f} €**"
            )

    else:
        st.info("Aucun canal n’a de conversions (selon les filtres actuels), impossible de calculer le CPA.")
else:
    st.info("Pas de données campagnes pour les filtres actuels.")


# -----------------------
# Conclusion (intégrée, actionnable)
# -----------------------
st.subheader("Conclusion opérationnelle")
st.markdown(
    """
- **Optimisation budget :** prioriser les canaux au **CPA bas** et **CTR élevé**, tout en surveillant la qualité (SQL/Client).
- **Pilotage funnel :** différencier les objectifs par canal (**volume** vs **qualité**).
- **Exécution :** renforcer l’approche **mobile-first** et ajuster le ciblage sur les **régions les plus porteuses**.
"""
)

# -----------------------
# Détails repliables (ne pas surcharger)
# -----------------------
with st.expander("Voir les données filtrées (détails)"):
    st.write("Leads filtrés")
    show_cols = [c for c in ["lead_id", "date", "channel", "device", "company_size", "sector", "region", "status"] if c in df.columns]
    st.dataframe(df[show_cols].sort_values("date"), use_container_width=True, hide_index=True)

    st.write("KPI par canal (campagnes)")
    if len(kpi):
        show_kpi = kpi[["channel", "cost", "impressions", "clicks", "conversions", "ctr", "cpa"]].copy()
        show_kpi["ctr"] = (show_kpi["ctr"] * 100).round(2)
        show_kpi["cpa"] = show_kpi["cpa"].round(2)
        show_kpi = show_kpi.rename(columns={"ctr": "ctr_%", "cpa": "cpa_€"})
        st.dataframe(show_kpi.sort_values("channel"), use_container_width=True, hide_index=True)
    else:
        st.write("Aucune donnée campagne pour les filtres actuels.")
