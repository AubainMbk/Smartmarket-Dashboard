import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Dashboard Marketing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# Style (compact + cartes homogènes)
# -----------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
      [data-testid="stMetricValue"] { font-size: 1.25rem; }
      [data-testid="stMetricLabel"] { font-size: 0.95rem; }
      div[data-testid="stVerticalBlock"] > div { gap: 0.7rem; }
      /* Réduit légèrement les titres des sous-sections */
      h3 { margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Constantes de rendu (homogénéité)
# -----------------------
FIGSIZE = (6.2, 3.35)
DPI = 140

CHANNEL_ORDER = ["Emailing", "Facebook Ads", "Instagram Ads", "LinkedIn"]
STATUS_ORDER = ["MQL", "SQL", "Client"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -----------------------
# Helpers
# -----------------------
def safe_div(a, b):
    return (a / b) if b not in (0, None) and not pd.isna(b) else np.nan

def norm_cols(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    return df

def strip_categories(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def card(title: str):
    """Simple wrapper pour uniformiser les blocs de graph."""
    c = st.container(border=True)
    with c:
        st.subheader(title)
    return c

# -----------------------
# Chargement
# -----------------------
@st.cache_data
def load_data():
    leads_path = DATA_DIR / "leads_smartmarket_extended.csv"
    crm_path = DATA_DIR / "crm_smartmarket_extended.xlsx"
    campaign_path = DATA_DIR / "campaign_smartmarket.json"

    missing = [str(p) for p in [leads_path, crm_path, campaign_path] if not p.exists()]
    if missing:
        st.error("Fichiers manquants :\n- " + "\n- ".join(missing))
        st.stop()

    leads = pd.read_csv(leads_path, parse_dates=["date"])
    crm = pd.read_excel(crm_path)
    with open(campaign_path, "r", encoding="utf-8") as f:
        campaigns = pd.DataFrame(json.load(f))

    leads = norm_cols(leads)
    crm = norm_cols(crm)
    campaigns = norm_cols(campaigns)

    # Périmètre Septembre 2025
    leads = leads[(leads["date"] >= "2025-09-01") & (leads["date"] <= "2025-09-30")].copy()

    # Nettoyage catégories (anti " IdF", " MQL", etc.)
    leads = strip_categories(leads, ["channel", "device"])
    crm = strip_categories(crm, ["region", "status", "company_size", "sector", "city"])
    campaigns = strip_categories(campaigns, ["channel"])

    # Jointure CRM
    leads_enriched = leads.merge(crm, on="lead_id", how="left", validate="one_to_one")
    return leads_enriched, campaigns

leads_enriched, campaigns = load_data()

# -----------------------
# Sidebar filtres
# -----------------------
st.sidebar.header("Filtres")

min_date = leads_enriched["date"].min()
max_date = leads_enriched["date"].max()

date_range = st.sidebar.date_input(
    "Période",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

all_channels = sorted(leads_enriched["channel"].dropna().unique().tolist())
all_regions = sorted(leads_enriched["region"].dropna().unique().tolist()) if "region" in leads_enriched.columns else []
all_status = sorted(leads_enriched["status"].dropna().unique().tolist()) if "status" in leads_enriched.columns else []
all_devices = sorted(leads_enriched["device"].dropna().unique().tolist()) if "device" in leads_enriched.columns else []

sel_channels = st.sidebar.multiselect("Canaux", options=all_channels, default=all_channels)
sel_regions = st.sidebar.multiselect("Régions", options=all_regions, default=all_regions) if all_regions else []
sel_status = st.sidebar.multiselect("Statut CRM", options=all_status, default=all_status) if all_status else []
sel_devices = st.sidebar.multiselect("Device", options=all_devices, default=all_devices) if all_devices else []

# Application filtres
df = leads_enriched.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d = pd.to_datetime(date_range[0])
    end_d = pd.to_datetime(date_range[1])
    df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]

if sel_channels:
    df = df[df["channel"].isin(sel_channels)]
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_status:
    df = df[df["status"].isin(sel_status)]
if sel_devices:
    df = df[df["device"].isin(sel_devices)]

camp = campaigns.copy()
if sel_channels:
    camp = camp[camp["channel"].isin(sel_channels)]

# -----------------------
# KPI globaux
# -----------------------
total_impr = camp["impressions"].sum() if len(camp) else 0
total_clicks = camp["clicks"].sum() if len(camp) else 0
total_conv = camp["conversions"].sum() if len(camp) else 0
total_cost = camp["cost"].sum() if len(camp) else 0

total_leads = df["lead_id"].nunique()

ctr_global = safe_div(total_clicks, total_impr)
cvr_postclick_global = safe_div(total_conv, total_clicks)
cpa_global = safe_div(total_cost, total_conv)
cpl_global = safe_div(total_cost, total_leads)

sql_count = (df["status"] == "SQL").sum() if "status" in df.columns else 0
client_count = (df["status"] == "Client").sum() if "status" in df.columns else 0
high_value_rate = safe_div((sql_count + client_count), total_leads)

# KPI canal (campagnes)
kpi = pd.DataFrame()
if len(camp):
    kpi = camp.copy()
    kpi["ctr"] = np.where(kpi["impressions"] > 0, kpi["clicks"] / kpi["impressions"], np.nan)
    kpi["cpa"] = np.where(kpi["conversions"] > 0, kpi["cost"] / kpi["conversions"], np.nan)

# Crosstab canal x statut
ct_channel_status = pd.DataFrame()
if "status" in df.columns and len(df):
    ct_channel_status = pd.crosstab(df["channel"], df["status"])

# -----------------------
# Header
# -----------------------
st.title("Dashboard Marketing (Septembre 2025)")
st.caption("Performance canaux, qualité CRM et focus géographique. Tous les indicateurs se recalculent via les filtres.")

# -----------------------
# Bandeau KPI
# -----------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("CTR global", f"{ctr_global*100:.2f}%" if not pd.isna(ctr_global) else "—")
k2.metric("Conv. post-clic", f"{cvr_postclick_global*100:.2f}%" if not pd.isna(cvr_postclick_global) else "—")
k3.metric("CPA global", f"{cpa_global:.2f} €" if not pd.isna(cpa_global) else "—")
k4.metric("CPL global", f"{cpl_global:.2f} €" if not pd.isna(cpl_global) else "—")
k5.metric("% SQL + Client", f"{high_value_rate*100:.1f}%" if not pd.isna(high_value_rate) else "—")

st.divider()

# -----------------------
# Grille 2x2 alignée (cartes)
# -----------------------
row1_left, row1_right = st.columns(2, gap="large")
row2_left, row2_right = st.columns(2, gap="large")

# ---- Graph 1 : CTR
with row1_left:
    with card("CTR par canal"):
        if len(kpi):
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            d = kpi.sort_values("ctr", ascending=False)
            ax.bar(d["channel"], d["ctr"])
            ax.set_title("CTR (clicks / impressions)", fontsize=11)
            ax.set_xlabel("Canal")
            ax.set_ylabel("CTR")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Pas de données campagnes pour les filtres actuels.")

# ---- Graph 2 : CPA
with row1_right:
    with card("CPA par canal"):
        if len(kpi):
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            d = kpi.sort_values("cpa", ascending=True)
            ax.bar(d["channel"], d["cpa"])
            ax.set_title("CPA (€) (cost / conversions)", fontsize=11)
            ax.set_xlabel("Canal")
            ax.set_ylabel("CPA (€)")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Pas de données campagnes pour les filtres actuels.")

# ---- Graph 3 : Qualité business
with row2_left:
    with card("Qualité business : Canal × Statut"):
        if not ct_channel_status.empty:
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

            ct = ct_channel_status.copy()

            # Ordre stable statuts
            cols = [c for c in STATUS_ORDER if c in ct.columns]
            if cols:
                ct = ct[cols]

            # Ordre stable canaux
            idx = [c for c in CHANNEL_ORDER if c in ct.index]
            if idx:
                ct = ct.reindex(idx)

            ct.plot(kind="bar", stacked=True, ax=ax, width=0.7)

            ax.set_title("Répartition des statuts par canal", fontsize=11)
            ax.set_xlabel("Canal")
            ax.set_ylabel("Nombre de leads")
            ax.tick_params(axis="x", rotation=20)

            # Légende en bas (évite décalage, + look propre)
            ax.legend(
                title="Statut",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=min(3, len(ct.columns)),
                frameon=False,
            )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Statut CRM indisponible ou aucun lead après filtrage.")

# ---- Graph 4 : Région
with row2_right:
    with card("Leads par région"):
        if "region" in df.columns and len(df):
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            region_counts = df["region"].value_counts()

            # Option : limiter à top 8 pour lisibilité
            top_n = 8
            region_counts = region_counts.head(top_n)

            ax.bar(region_counts.index.astype(str), region_counts.values)
            ax.set_title("Volume de leads par région (Top régions)", fontsize=11)
            ax.set_xlabel("Région")
            ax.set_ylabel("Nombre de leads")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Région indisponible ou aucun lead après filtrage.")

st.divider()

# -----------------------
# Insight dynamique (CPA)
# -----------------------
st.subheader("Insight dynamique (CPA)")

if len(kpi):
    kpi_valid = kpi.dropna(subset=["cpa"]).copy()
    if len(kpi_valid):
        best = kpi_valid.sort_values("cpa", ascending=True).iloc[0]
        worst = kpi_valid.sort_values("cpa", ascending=False).iloc[0]

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.success(
                f"**Meilleur canal (CPA minimal)** : **{best['channel']}**\n\n"
                f"- CPA : **{float(best['cpa']):.2f} €**\n"
                f"- Conversions : **{int(best['conversions'])}**\n"
                f"- Dépenses : **{float(best['cost']):.0f} €**"
            )

        with col_b:
            st.warning(
                f"**Canal à optimiser (CPA élevé)** : **{worst['channel']}**\n\n"
                f"- CPA : **{float(worst['cpa']):.2f} €**\n"
                f"- Conversions : **{int(worst['conversions'])}**\n"
                f"- Dépenses : **{float(worst['cost']):.0f} €**"
            )
    else:
        st.info("Aucun canal n’a de conversions (selon les filtres actuels), impossible de calculer le CPA.")
else:
    st.info("Pas de données campagnes pour les filtres actuels.")

# -----------------------
# Conclusion
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
# Détails (repliable)
# -----------------------
with st.expander("Voir les données filtrées (détails)"):
    st.write("Leads filtrés")
    show_cols = [c for c in ["lead_id", "date", "channel", "device", "company_size", "sector", "region", "city", "status"] if c in df.columns]
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
