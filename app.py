
import io
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Google Ads Funnel EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

REQUIRED_COLUMNS = [
    "Fecha",
    "Campaign_ID",
    "Campana",
    "Impresiones",
    "Clics",
    "Inversion",
    "Leads",
]

OPTIONAL_FILTER_COLUMNS = [
    "Grupo_de_Anuncios",
    "Tipo_Campana",
    "Estado_Anuncio",
    "Ad_ID",
    "Anuncio",
]

# Campos que NO se usan como fuente de verdad porque se recalculan dinámicamente.
DERIVED_COLUMNS_TO_IGNORE = [
    "CTR",
    "CPC",
    "CPM",
    "Costo_por_Lead",
    "CAC",
    "ROAS",
    "Pct_Calificacion",
    "Costo_por_Calificado",
]


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(ttl=600)
def load_from_public_csv(csv_url: str) -> pd.DataFrame:
    """Carga un Google Sheet mediante URL CSV/export o cualquier CSV accesible."""
    return pd.read_csv(csv_url)


@st.cache_data(ttl=600)
def load_from_private_google_sheet(
    spreadsheet_url: str,
    worksheet_name: str,
    service_account_info: dict,
) -> pd.DataFrame:
    """
    Carga una hoja privada de Google Sheets con gspread.
    Requiere compartir el Sheet con el client_email del service account.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes,
    )

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(spreadsheet_url)

    if worksheet_name.strip():
        ws = sh.worksheet(worksheet_name.strip())
    else:
        ws = sh.sheet1

    records = ws.get_all_records()
    return pd.DataFrame(records)


def load_data() -> pd.DataFrame:
    """
    Orden de prioridad:
    1) Google Sheet privado mediante st.secrets
    2) URL CSV mediante st.secrets
    3) Carga manual CSV
    """
    private_config_ok = (
        "gsheets" in st.secrets
        and "spreadsheet_url" in st.secrets["gsheets"]
        and "gcp_service_account" in st.secrets
    )

    if private_config_ok:
        try:
            worksheet = st.secrets["gsheets"].get("worksheet_name", "")
            return load_from_private_google_sheet(
                st.secrets["gsheets"]["spreadsheet_url"],
                worksheet,
                dict(st.secrets["gcp_service_account"]),
            )
        except Exception as e:
            st.sidebar.error(f"No se pudo leer el Google Sheet privado: {e}")

    if "GOOGLE_SHEET_CSV_URL" in st.secrets:
        try:
            return load_from_public_csv(st.secrets["GOOGLE_SHEET_CSV_URL"])
        except Exception as e:
            st.sidebar.error(f"No se pudo leer GOOGLE_SHEET_CSV_URL: {e}")

    st.sidebar.info(
        "No encontré configuración de Google Sheets en st.secrets. "
        "Puedes cargar un CSV temporalmente."
    )
    uploaded = st.sidebar.file_uploader("Cargar CSV", type=["csv"])

    if uploaded is None:
        st.info(
            "Configura tu Google Sheet en `.streamlit/secrets.toml` "
            "o carga un CSV desde la barra lateral."
        )
        st.stop()

    return pd.read_csv(uploaded)


# ============================================================
# LIMPIEZA Y PREPARACIÓN
# ============================================================

def normalize_colname(col: str) -> str:
    return str(col).strip()


def parse_number(series: pd.Series) -> pd.Series:
    """
    Convierte números que pueden venir como:
    20.83
    "$20.83"
    "1,234.50"
    "9.30%"
    ""
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    s = s.str.replace(r"[\$,€£%]", "", regex=True)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(
            "Faltan columnas obligatorias: "
            + ", ".join(missing)
            + ". Revisa los nombres de tu Google Sheet."
        )
        st.stop()

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    numeric_cols = [
        "Impresiones",
        "Clics",
        "Inversion",
        "Leads",
        "Campaign_ID",
        "Ad_ID",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_number(df[col])

    # Conservamos las columnas calculadas originales si existen,
    # pero NO se usan para KPIs o agregaciones.
    for col in DERIVED_COLUMNS_TO_IGNORE:
        if col in df.columns:
            df[f"{col}_Original"] = df[col]

    df = df.dropna(subset=["Fecha"]).copy()

    # Evita NaN en métricas base.
    for col in ["Impresiones", "Clics", "Inversion", "Leads"]:
        df[col] = df[col].fillna(0.0)

    # Variables temporales
    df["Dia"] = df["Fecha"].dt.date
    df["Mes"] = df["Fecha"].dt.to_period("M").astype(str)
    month_map = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
    }
    df["Mes_Orden"] = df["Fecha"].dt.to_period("M").astype(str)
    df["Mes_Label"] = df.apply(
        lambda r: f"{month_map[r['Fecha'].month]} {r['Fecha'].year}",
        axis=1,
    )
    df["Semana"] = df["Fecha"].dt.to_period("W-MON").apply(
        lambda p: p.start_time.date()
    )
    df["Dia_Semana_Num"] = df["Fecha"].dt.dayofweek

    day_map = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }
    df["Dia_Semana"] = df["Dia_Semana_Num"].map(day_map)

    return df


# ============================================================
# MÉTRICAS
# ============================================================

def safe_div(num, den):
    if den is None or den == 0 or pd.isna(den):
        return np.nan
    return num / den


def aggregate_metrics(df: pd.DataFrame) -> dict:
    impressions = df["Impresiones"].sum()
    clicks = df["Clics"].sum()
    leads = df["Leads"].sum()
    spend = df["Inversion"].sum()

    return {
        "Impresiones": impressions,
        "Clics": clicks,
        "Leads": leads,
        "Inversion": spend,
        "CTR": safe_div(clicks, impressions),
        "CPC": safe_div(spend, clicks),
        "CPM": safe_div(spend * 1000, impressions),
        "CPL": safe_div(spend, leads),
        "CVR_Click_Lead": safe_div(leads, clicks),
        "CVR_Imp_Lead": safe_div(leads, impressions),
    }


def grouped_metrics(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, dropna=False, observed=False)[
            ["Impresiones", "Clics", "Leads", "Inversion"]
        ]
        .sum()
        .reset_index()
    )

    out["CTR"] = out["Clics"] / out["Impresiones"].replace(0, np.nan)
    out["CPC"] = out["Inversion"] / out["Clics"].replace(0, np.nan)
    out["CPM"] = (
        out["Inversion"] * 1000 / out["Impresiones"].replace(0, np.nan)
    )
    out["CPL"] = out["Inversion"] / out["Leads"].replace(0, np.nan)
    out["CVR_Click_Lead"] = (
        out["Leads"] / out["Clics"].replace(0, np.nan)
    )
    out["CVR_Imp_Lead"] = (
        out["Leads"] / out["Impresiones"].replace(0, np.nan)
    )

    return out


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["Impresiones", "Clics", "Leads"]:
        if col in out.columns:
            out[col] = out[col].round(0).astype("Int64")

    for col in ["Inversion", "CPC", "CPM", "CPL"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"${x:,.2f}"
            )

    for col in ["CTR", "CVR_Click_Lead", "CVR_Imp_Lead"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.2%}"
            )

    return out


def funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    m = aggregate_metrics(df)

    stages = [
        {
            "Etapa": "Impresiones",
            "Volumen": m["Impresiones"],
            "Conv_etapa_anterior": 1.0,
            "Conv_acumulada_desde_impresion": 1.0,
            "Metrica_Costo": "—",
            "Costo": np.nan,
        },
        {
            "Etapa": "Clics",
            "Volumen": m["Clics"],
            "Conv_etapa_anterior": m["CTR"],
            "Conv_acumulada_desde_impresion": m["CTR"],
            "Metrica_Costo": "CPC",
            "Costo": m["CPC"],
        },
        {
            "Etapa": "Leads",
            "Volumen": m["Leads"],
            "Conv_etapa_anterior": m["CVR_Click_Lead"],
            "Conv_acumulada_desde_impresion": m["CVR_Imp_Lead"],
            "Metrica_Costo": "CPL",
            "Costo": m["CPL"],
        },
    ]

    return pd.DataFrame(stages)


def monthly_funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    out = grouped_metrics(df, ["Mes_Orden", "Mes_Label", "Campana"])
    if out.empty:
        return out

    out = out.sort_values(
        ["Mes_Orden", "Campana"],
        ascending=[False, True],
    )

    cols = [
        "Mes_Label",
        "Campana",
        "Impresiones",
        "Clics",
        "Leads",
        "CTR",
        "CVR_Click_Lead",
        "CVR_Imp_Lead",
    ]
    return out[cols]


def monthly_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    out = grouped_metrics(df, ["Mes_Orden", "Mes_Label", "Campana"])
    if out.empty:
        return out

    out = out.sort_values(
        ["Mes_Orden", "Campana"],
        ascending=[False, True],
    )

    cols = [
        "Mes_Label",
        "Campana",
        "Inversion",
        "CPC",
        "CPL",
    ]
    return out[cols]


# ============================================================
# COMPONENTES VISUALES
# ============================================================

def kpi_row(df: pd.DataFrame):
    m = aggregate_metrics(df)

    cols = st.columns(7)

    cols[0].metric("Inversión", f"${m['Inversion']:,.2f}")
    cols[1].metric("Impresiones", f"{m['Impresiones']:,.0f}")
    cols[2].metric("Clics", f"{m['Clics']:,.0f}")
    cols[3].metric("Leads", f"{m['Leads']:,.0f}")
    cols[4].metric(
        "CTR",
        "—" if pd.isna(m["CTR"]) else f"{m['CTR']:.2%}",
    )
    cols[5].metric(
        "CPC",
        "—" if pd.isna(m["CPC"]) else f"${m['CPC']:,.2f}",
    )
    cols[6].metric(
        "CPL",
        "—" if pd.isna(m["CPL"]) else f"${m['CPL']:,.2f}",
    )


def funnel_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()

    campaigns = (
        df["Campana"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    for campaign in campaigns:
        campaign_df = df[df["Campana"].astype(str) == campaign]
        ft = funnel_table(campaign_df)

        fig.add_trace(
            go.Funnel(
                name=campaign,
                y=ft["Etapa"],
                x=ft["Volumen"],
                textinfo="value+percent initial+percent previous",
                hovertemplate=(
                    f"<b>{campaign}</b><br>"
                    "Etapa: %{y}<br>"
                    "Volumen: %{x:,.0f}<br>"
                    "% del inicio: %{percentInitial:.2%}<br>"
                    "% etapa previa: %{percentPrevious:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        funnelmode="group",
        margin=dict(l=20, r=20, t=60, b=20),
        height=470,
        legend_title_text="Campaña",
    )

    return fig


def cost_stage_chart(df: pd.DataFrame, title: str):
    m = aggregate_metrics(df)

    c = pd.DataFrame(
        {
            "Métrica": ["CPC", "CPL"],
            "Costo": [m["CPC"], m["CPL"]],
            "Significado": [
                "Costo por clic",
                "Costo por lead",
            ],
        }
    )

    fig = px.bar(
        c,
        x="Métrica",
        y="Costo",
        text_auto=".2f",
        hover_data=["Significado"],
        title=title,
    )

    fig.update_layout(
        yaxis_title="Costo",
        xaxis_title="",
        height=430,
    )

    return fig


def make_timeseries(
    df: pd.DataFrame,
    metric: str,
    granularity: str,
    by_campaign: bool,
):
    freq_map = {
        "Día": "D",
        "Semana": "W-MON",
        "Mes": "MS",
    }
    freq = freq_map[granularity]

    tmp = df.copy()
    tmp["Periodo"] = tmp["Fecha"].dt.to_period(freq).dt.start_time

    group_cols = ["Periodo"]
    if by_campaign:
        group_cols.append("Campana")

    daily = grouped_metrics(tmp, group_cols)

    fig = px.line(
        daily,
        x="Periodo",
        y=metric,
        color="Campana" if by_campaign else None,
        markers=True,
        title=f"{metric} por {granularity.lower()}",
    )
    fig.update_layout(height=430)

    return fig, daily


def descriptive_stats_on_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = grouped_metrics(df.assign(Dia=df["Fecha"].dt.date), "Dia")
    if daily.empty:
        return pd.DataFrame()

    metrics = [
        "Impresiones",
        "Clics",
        "Leads",
        "Inversion",
        "CTR",
        "CPC",
        "CPM",
        "CPL",
        "CVR_Click_Lead",
    ]

    rows = []

    for metric in metrics:
        s = pd.to_numeric(daily[metric], errors="coerce").dropna()

        if s.empty:
            continue

        mean = s.mean()
        sd = s.std(ddof=1) if len(s) > 1 else 0.0
        median = s.median()
        cv = safe_div(sd, mean)

        rows.append(
            {
                "Métrica": metric,
                "N_días": len(s),
                "Promedio": mean,
                "Mediana": median,
                "SD": sd,
                "CV": cv,
                "Mínimo": s.min(),
                "Máximo": s.max(),
            }
        )

    return pd.DataFrame(rows)


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = grouped_metrics(
        df,
        ["Dia_Semana_Num", "Dia_Semana"],
    )

    if out.empty:
        return out

    active_days = (
        df.groupby(["Dia_Semana_Num", "Dia_Semana"])["Fecha"]
        .apply(lambda s: s.dt.date.nunique())
        .reset_index(name="Dias_Observados")
    )

    out = out.merge(
        active_days,
        on=["Dia_Semana_Num", "Dia_Semana"],
        how="left",
    )

    out["Leads_por_Dia"] = (
        out["Leads"] / out["Dias_Observados"].replace(0, np.nan)
    )
    out["Clicks_por_Dia"] = (
        out["Clics"] / out["Dias_Observados"].replace(0, np.nan)
    )
    out["Inversion_por_Dia"] = (
        out["Inversion"] / out["Dias_Observados"].replace(0, np.nan)
    )

    return out.sort_values("Dia_Semana_Num")


def daily_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    daily = grouped_metrics(
        df.assign(Dia=df["Fecha"].dt.date),
        "Dia",
    )

    if daily.empty:
        return daily

    # Estadística descriptiva: distancia en SD respecto al promedio.
    for metric in ["Leads", "Clics", "CTR", "CPL", "Inversion"]:
        s = pd.to_numeric(daily[metric], errors="coerce")
        sd = s.std(ddof=1)
        mean = s.mean()

        if pd.isna(sd) or sd == 0:
            daily[f"Z_{metric}"] = np.nan
        else:
            daily[f"Z_{metric}"] = (s - mean) / sd

    def label_row(row):
        labels = []

        if pd.notna(row.get("Z_Leads")) and row["Z_Leads"] <= -1:
            labels.append("Leads bajos vs. promedio")

        if pd.notna(row.get("Z_CPL")) and row["Z_CPL"] >= 1:
            labels.append("CPL alto vs. promedio")

        if pd.notna(row.get("Z_CTR")) and row["Z_CTR"] <= -1:
            labels.append("CTR bajo vs. promedio")

        if pd.notna(row.get("Z_Clics")) and row["Z_Clics"] <= -1:
            labels.append("Clics bajos vs. promedio")

        return "; ".join(labels) if labels else "Dentro del rango descriptivo"

    daily["Lectura_Descriptiva"] = daily.apply(label_row, axis=1)
    return daily.sort_values("Dia", ascending=False)


def correlation_matrix(df: pd.DataFrame):
    daily = grouped_metrics(
        df.assign(Dia=df["Fecha"].dt.date),
        "Dia",
    )

    cols = [
        "Impresiones",
        "Clics",
        "Leads",
        "Inversion",
        "CTR",
        "CPC",
        "CPM",
        "CPL",
        "CVR_Click_Lead",
    ]

    corr = daily[cols].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title="Correlación descriptiva entre métricas diarias",
    )
    fig.update_layout(height=600)

    return fig, corr


def period_summary(df: pd.DataFrame, label: str) -> dict:
    m = aggregate_metrics(df)
    return {
        "Periodo": label,
        **m,
    }


# ============================================================
# APP
# ============================================================

st.title("Google Ads Funnel EDA")

raw_df = load_data()
df = clean_data(raw_df)

# ------------------------------------------------------------
# FILTROS GLOBALES
# ------------------------------------------------------------

st.sidebar.header("Filtros globales")

min_date = df["Fecha"].min().date()
max_date = df["Fecha"].max().date()

selected_dates = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = selected_dates
    end_date = selected_dates

filtered = df[
    (df["Fecha"].dt.date >= start_date)
    & (df["Fecha"].dt.date <= end_date)
].copy()

# Filtros categóricos
for col in [
    "Campana",
    "Grupo_de_Anuncios",
    "Tipo_Campana",
    "Estado_Anuncio",
]:
    if col in filtered.columns:
        values = (
            filtered[col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        chosen = st.sidebar.multiselect(
            col.replace("_", " "),
            options=values,
            default=values,
        )

        if chosen:
            filtered = filtered[
                filtered[col].astype(str).isin(chosen)
            ]
        else:
            filtered = filtered.iloc[0:0]

st.sidebar.divider()
st.sidebar.caption(
    f"Filas activas: {len(filtered):,} | "
    f"{start_date} a {end_date}"
)

if filtered.empty:
    st.warning("No hay datos para la combinación actual de filtros.")
    st.stop()

# Las dimensiones filtradas se conservan para el comparador,
# pero los periodos A/B pueden ser distintos al rango de la pestaña 1.
selected_campaigns = filtered["Campana"].dropna().unique().tolist()

tab1, tab2 = st.tabs(
    ["Funnel Adwords", "Análisis del funnel GA"]
)

# ============================================================
# TAB 1 — FUNNEL ADWORDS
# ============================================================

with tab1:
    st.subheader("1. Funnel Adwords")
    kpi_row(filtered)

    st.divider()

    funnel_df = filtered.copy()
    funnel_title = "Funnel por campaña"

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.plotly_chart(
            funnel_chart(funnel_df, funnel_title),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            cost_stage_chart(
                funnel_df,
                "Costo por etapa del funnel",
            ),
            use_container_width=True,
        )

    st.markdown("#### Detalle del funnel seleccionado")

    ft = funnel_table(funnel_df).copy()
    ft["Volumen"] = ft["Volumen"].round(0).astype("Int64")
    ft["Conv_etapa_anterior"] = ft["Conv_etapa_anterior"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2%}"
    )
    ft["Conv_acumulada_desde_impresion"] = (
        ft["Conv_acumulada_desde_impresion"].map(
            lambda x: "" if pd.isna(x) else f"{x:.2%}"
        )
    )
    ft["Costo"] = ft["Costo"].map(
        lambda x: "" if pd.isna(x) else f"${x:,.2f}"
    )

    st.dataframe(
        ft,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("### Tasas por mes y campaña")
    monthly_funnel = monthly_funnel_table(filtered)
    st.dataframe(
        format_metrics_table(monthly_funnel),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Costos por mes y campaña")
    monthly_cost = monthly_cost_table(filtered)
    st.dataframe(
        format_metrics_table(monthly_cost),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Resumen por campaña")
    by_campaign = grouped_metrics(filtered, "Campana")
    by_campaign = by_campaign.drop(columns=["CPM"], errors="ignore")
    by_campaign = by_campaign.sort_values(
        "Leads",
        ascending=False,
    )
    st.dataframe(
        format_metrics_table(by_campaign),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Datos filtrados")
    st.dataframe(
        filtered.sort_values("Fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 2 — ANÁLISIS EXPLORATORIO
# ============================================================

with tab2:
    st.subheader("2. Análisis del funnel GA")
    st.caption(
        "EDA descriptivo para localizar patrones antes de modelar: "
        "tendencias, variabilidad, días débiles, relación entre métricas "
        "y comparación de periodos."
    )

    # --------------------------------------------------------
    # COMPARADOR A/B
    # --------------------------------------------------------
    st.markdown("### Comparador de periodos")

    base_filtered = df.copy()

    # Respeta los filtros categóricos activos de la pestaña 1.
    # Para ello tomamos las categorías que sobrevivieron a los filtros.
    for col in [
        "Campana",
        "Grupo_de_Anuncios",
        "Tipo_Campana",
        "Estado_Anuncio",
    ]:
        if col in filtered.columns and col in base_filtered.columns:
            allowed = filtered[col].dropna().astype(str).unique().tolist()
            if allowed:
                base_filtered = base_filtered[
                    base_filtered[col].astype(str).isin(allowed)
                ]

    data_min = base_filtered["Fecha"].min().date()
    data_max = base_filtered["Fecha"].max().date()

    c1, c2 = st.columns(2)

    with c1:
        period_a = st.date_input(
            "Periodo A",
            value=(start_date, end_date),
            min_value=data_min,
            max_value=data_max,
            key="period_a",
        )

    with c2:
        # Por defecto intenta usar el periodo inmediatamente anterior
        period_days = (end_date - start_date).days + 1
        default_b_end = start_date - pd.Timedelta(days=1)
        default_b_start = default_b_end - pd.Timedelta(
            days=period_days - 1
        )

        default_b_start = max(
            pd.Timestamp(data_min),
            pd.Timestamp(default_b_start),
        ).date()

        default_b_end = max(
            pd.Timestamp(data_min),
            pd.Timestamp(default_b_end),
        ).date()

        period_b = st.date_input(
            "Periodo B",
            value=(default_b_start, default_b_end),
            min_value=data_min,
            max_value=data_max,
            key="period_b",
        )

    def unpack_period(period_value):
        if isinstance(period_value, tuple) and len(period_value) == 2:
            return period_value[0], period_value[1]
        return period_value, period_value

    a_start, a_end = unpack_period(period_a)
    b_start, b_end = unpack_period(period_b)

    df_a = base_filtered[
        (base_filtered["Fecha"].dt.date >= a_start)
        & (base_filtered["Fecha"].dt.date <= a_end)
    ].copy()

    df_b = base_filtered[
        (base_filtered["Fecha"].dt.date >= b_start)
        & (base_filtered["Fecha"].dt.date <= b_end)
    ].copy()

    compare = pd.DataFrame(
        [
            period_summary(
                df_a,
                f"A: {a_start} → {a_end}",
            ),
            period_summary(
                df_b,
                f"B: {b_start} → {b_end}",
            ),
        ]
    )

    st.dataframe(
        format_metrics_table(compare),
        use_container_width=True,
        hide_index=True,
    )

    if not df_a.empty and not df_b.empty:
        ma = aggregate_metrics(df_a)
        mb = aggregate_metrics(df_b)

        delta_rows = []
        for metric in [
            "Impresiones",
            "Clics",
            "Leads",
            "Inversion",
            "CTR",
            "CPC",
            "CPM",
            "CPL",
            "CVR_Click_Lead",
        ]:
            a_val = ma[metric]
            b_val = mb[metric]
            delta = safe_div(a_val - b_val, b_val)

            delta_rows.append(
                {
                    "Métrica": metric,
                    "Periodo_A": a_val,
                    "Periodo_B": b_val,
                    "Cambio_relativo_A_vs_B": delta,
                }
            )

        delta_df = pd.DataFrame(delta_rows)

        def fmt_compare_value(row, col):
            value = row[col]
            metric = row["Métrica"]

            if pd.isna(value):
                return ""

            if metric in [
                "CTR",
                "CVR_Click_Lead",
            ]:
                return f"{value:.2%}"

            if metric in [
                "Inversion",
                "CPC",
                "CPM",
                "CPL",
            ]:
                return f"${value:,.2f}"

            return f"{value:,.0f}"

        delta_display = delta_df.copy()
        delta_display["Periodo_A"] = delta_display.apply(
            lambda r: fmt_compare_value(r, "Periodo_A"),
            axis=1,
        )
        delta_display["Periodo_B"] = delta_display.apply(
            lambda r: fmt_compare_value(r, "Periodo_B"),
            axis=1,
        )
        delta_display["Cambio_relativo_A_vs_B"] = (
            delta_display["Cambio_relativo_A_vs_B"].map(
                lambda x: "" if pd.isna(x) else f"{x:+.2%}"
            )
        )

        st.dataframe(
            delta_display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --------------------------------------------------------
    # SERIES DE TIEMPO
    # --------------------------------------------------------
    st.markdown("### Series de tiempo")

    ts1, ts2, ts3 = st.columns(3)

    with ts1:
        metric = st.selectbox(
            "Métrica",
            options=[
                "Impresiones",
                "Clics",
                "Leads",
                "Inversion",
                "CTR",
                "CPC",
                "CPM",
                "CPL",
                "CVR_Click_Lead",
            ],
            index=2,
        )

    with ts2:
        granularity = st.selectbox(
            "Granularidad",
            options=["Día", "Semana", "Mes"],
            index=0,
        )

    with ts3:
        by_campaign = st.checkbox(
            "Separar por campaña",
            value=False,
        )

    ts_fig, ts_table = make_timeseries(
        filtered,
        metric,
        granularity,
        by_campaign,
    )

    st.plotly_chart(
        ts_fig,
        use_container_width=True,
    )

    st.dataframe(
        format_metrics_table(ts_table),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # PERFIL POR DÍA DE LA SEMANA
    # --------------------------------------------------------
    st.markdown("### Comportamiento por día de la semana")

    weekday = weekday_profile(filtered)

    wc1, wc2 = st.columns(2)

    with wc1:
        fig = px.bar(
            weekday,
            x="Dia_Semana",
            y="Leads_por_Dia",
            category_orders={
                "Dia_Semana": [
                    "Lunes",
                    "Martes",
                    "Miércoles",
                    "Jueves",
                    "Viernes",
                    "Sábado",
                    "Domingo",
                ]
            },
            title="Leads promedio por día observado",
            text_auto=".2f",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with wc2:
        fig = px.bar(
            weekday,
            x="Dia_Semana",
            y="CPL",
            category_orders={
                "Dia_Semana": [
                    "Lunes",
                    "Martes",
                    "Miércoles",
                    "Jueves",
                    "Viernes",
                    "Sábado",
                    "Domingo",
                ]
            },
            title="CPL por día de la semana",
            text_auto=".2f",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    weekday_display = weekday[
        [
            "Dia_Semana",
            "Dias_Observados",
            "Impresiones",
            "Clics",
            "Leads",
            "Leads_por_Dia",
            "Inversion",
            "CTR",
            "CPC",
            "CPL",
            "CVR_Click_Lead",
        ]
    ].copy()

    st.dataframe(
        format_metrics_table(weekday_display),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # ESTADÍSTICA DESCRIPTIVA
    # --------------------------------------------------------
    st.markdown("### Estadística descriptiva diaria")

    stats = descriptive_stats_on_daily(filtered)

    stats_display = stats.copy()

    for col in ["Promedio", "Mediana", "SD", "Mínimo", "Máximo"]:
        stats_display[col] = stats_display[col].map(
            lambda x: "" if pd.isna(x) else f"{x:,.4f}"
        )

    stats_display["CV"] = stats_display["CV"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2%}"
    )

    st.dataframe(
        stats_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "CV = desviación estándar / promedio. "
        "Un CV alto indica mayor variabilidad relativa; úsalo como señal "
        "para decidir qué métrica merece un análisis posterior."
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELACIONES DESCRIPTIVAS
    # --------------------------------------------------------
    st.markdown("### Relaciones entre métricas")

    corr_fig, corr_table = correlation_matrix(filtered)

    st.plotly_chart(
        corr_fig,
        use_container_width=True,
    )

    st.dataframe(
        corr_table.round(3),
        use_container_width=True,
    )

    st.caption(
        "La correlación es exploratoria: sirve para detectar relaciones "
        "que después puedes validar con modelos. No implica causalidad."
    )

    st.divider()

    # --------------------------------------------------------
    # DISPERSIÓN EXPLORATORIA
    # --------------------------------------------------------
    st.markdown("### Explorador de dispersión")

    daily = grouped_metrics(
        filtered.assign(Dia=filtered["Fecha"].dt.date),
        "Dia",
    )

    sc1, sc2 = st.columns(2)

    scatter_options = [
        "Impresiones",
        "Clics",
        "Leads",
        "Inversion",
        "CTR",
        "CPC",
        "CPM",
        "CPL",
        "CVR_Click_Lead",
    ]

    with sc1:
        scatter_x = st.selectbox(
            "Eje X",
            scatter_options,
            index=3,
        )

    with sc2:
        scatter_y = st.selectbox(
            "Eje Y",
            scatter_options,
            index=2,
        )

    scatter_fig = px.scatter(
        daily,
        x=scatter_x,
        y=scatter_y,
        hover_data=["Dia"],
        size="Impresiones",
        title=f"{scatter_x} vs {scatter_y} por día",
    )
    scatter_fig.update_layout(height=450)

    st.plotly_chart(
        scatter_fig,
        use_container_width=True,
    )

    st.dataframe(
        format_metrics_table(daily.sort_values("Dia", ascending=False)),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # DÍAS DÉBILES / DIAGNÓSTICOS DESCRIPTIVOS
    # --------------------------------------------------------
    st.markdown("### Días débiles y señales descriptivas")

    diagnostics = daily_diagnostics(filtered)

    interesting = diagnostics[
        diagnostics["Lectura_Descriptiva"]
        != "Dentro del rango descriptivo"
    ].copy()

    if interesting.empty:
        st.info(
            "No aparecen días con desviaciones descriptivas de al menos "
            "1 SD en Leads, CPL, CTR o Clics dentro del rango filtrado."
        )
    else:
        show_cols = [
            "Dia",
            "Impresiones",
            "Clics",
            "Leads",
            "Inversion",
            "CTR",
            "CPC",
            "CPL",
            "CVR_Click_Lead",
            "Z_Leads",
            "Z_CPL",
            "Z_CTR",
            "Lectura_Descriptiva",
        ]

        diag_display = interesting[show_cols].copy()
        diag_display = format_metrics_table(diag_display)

        for col in ["Z_Leads", "Z_CPL", "Z_CTR"]:
            if col in diag_display.columns:
                diag_display[col] = pd.to_numeric(
                    interesting[col],
                    errors="coerce",
                ).map(
                    lambda x: "" if pd.isna(x) else f"{x:.2f}"
                )

        st.dataframe(
            diag_display,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Estas etiquetas no son un modelo ni una prueba estadística. "
        "Sólo marcan días que se alejan del promedio del periodo filtrado "
        "para facilitar la inspección."
    )

    # --------------------------------------------------------
    # OPCIONAL: PERFIL POR HORA SI EXISTE COLUMNA
    # --------------------------------------------------------
    hour_col = None
    for candidate in ["Hora", "Hour", "Hora_del_dia", "Hora del día"]:
        if candidate in filtered.columns:
            hour_col = candidate
            break

    if hour_col is not None:
        st.divider()
        st.markdown("### Perfil por hora")

        hour_df = filtered.copy()
        hour_df[hour_col] = parse_number(hour_df[hour_col])

        hour_profile = grouped_metrics(
            hour_df.dropna(subset=[hour_col]),
            hour_col,
        ).sort_values(hour_col)

        hfig = px.line(
            hour_profile,
            x=hour_col,
            y="Leads",
            markers=True,
            title="Leads por hora",
        )
        st.plotly_chart(hfig, use_container_width=True)

        st.dataframe(
            format_metrics_table(hour_profile),
            use_container_width=True,
            hide_index=True,
        )
