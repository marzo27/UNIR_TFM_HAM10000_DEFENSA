"""Funciones puras (sin dependencia de ningún framework de UI) para comparar
dos modelos ya cargados con utils.carga_metricas.

Devuelven DataFrames de pandas y Figuras de Plotly listos para insertarse en
cualquier UI (Gradio, Streamlit, etc.). No recalculan ni inventan ninguna
métrica: solo reorganizan lo que ya devuelven cargar_svm / cargar_densenet.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.plot_utils import aplicar_layout_responsivo

ETIQUETAS_METRICAS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "F1 Macro",
    "f1_weighted": "F1 Weighted",
    "precision_macro": "Precision Macro",
    "recall_macro": "Recall Macro",
    "auc_macro": "AUC Macro",
    "cohen_kappa": "Cohen's Kappa",
    "mcc": "MCC",
}

METRICAS_KPI = ["accuracy", "f1_macro", "recall_macro", "auc_macro"]


def tabla_comparativa_global(modelo_a: dict, modelo_b: dict) -> pd.DataFrame:
    """Tabla Métrica | Modelo A | Modelo B, con la métrica como columna normal
    (no como índice) para que se muestre bien en gr.Dataframe."""
    filas = []
    for clave, etiqueta in ETIQUETAS_METRICAS.items():
        val_a = modelo_a["global"].get(clave)
        val_b = modelo_b["global"].get(clave)
        if val_a is None and val_b is None:
            continue
        filas.append({"Métrica": etiqueta, modelo_a["nombre"]: val_a, modelo_b["nombre"]: val_b})
    df = pd.DataFrame(filas)
    for col in df.columns[1:]:
        df[col] = df[col].astype(float).round(4)
    return df


def tabla_kpis_delta(modelo_a: dict, modelo_b: dict) -> pd.DataFrame:
    """Tabla resumen con el delta (Modelo B - Modelo A) para las métricas clave."""
    filas = []
    for clave in METRICAS_KPI:
        val_a = modelo_a["global"].get(clave)
        val_b = modelo_b["global"].get(clave)
        if val_a is None or val_b is None:
            filas.append({"Métrica": ETIQUETAS_METRICAS.get(clave, clave), modelo_a["nombre"]: val_a, modelo_b["nombre"]: val_b, "Δ (B - A)": None})
            continue
        filas.append(
            {
                "Métrica": ETIQUETAS_METRICAS.get(clave, clave),
                modelo_a["nombre"]: round(float(val_a), 4),
                modelo_b["nombre"]: round(float(val_b), 4),
                "Δ (B - A)": round(float(val_b) - float(val_a), 4),
            }
        )
    return pd.DataFrame(filas)


def grafico_barras_comparativo(modelo_a: dict, modelo_b: dict) -> go.Figure:
    df = tabla_comparativa_global(modelo_a, modelo_b)
    df_plot = df.melt(id_vars="Métrica", var_name="Modelo", value_name="Valor")
    fig = px.bar(
        df_plot, x="Métrica", y="Valor", color="Modelo", barmode="group",
        range_y=[0, 1], text_auto=".3f",
    )
    fig.update_layout(xaxis_tickangle=-30, margin=dict(t=30))
    return aplicar_layout_responsivo(fig)


def tabla_por_clase(modelo: dict) -> pd.DataFrame:
    df = pd.DataFrame(modelo["por_clase"]).T.round(4)
    df = df.reset_index().rename(columns={"index": "Clase"})
    return df


def graficos_por_clase_comparados(modelo_a: dict, modelo_b: dict, metrica: str = "f1_score") -> tuple[go.Figure | None, str | None]:
    """Compara una métrica por clase entre dos modelos.

    Empareja clases ignorando mayúsculas/minúsculas (p. ej. "Benigno" del SVM
    con "benigno" de DenseNet) sin alterar los nombres originales mostrados.
    Devuelve (figura, None) si hay clases comunes, o (None, mensaje_aviso) si no.
    """
    mapa_a = {c.lower(): c for c in modelo_a["clases"]}
    mapa_b = {c.lower(): c for c in modelo_b["clases"]}
    comunes = [k for k in mapa_a if k in mapa_b]
    if not comunes:
        aviso = (
            "Los modelos no comparten clases directamente comparables "
            f"({modelo_a['nombre']}: {modelo_a['clases']} vs "
            f"{modelo_b['nombre']}: {modelo_b['clases']})."
        )
        return None, aviso

    filas = []
    for clave in comunes:
        clase_a, clase_b = mapa_a[clave], mapa_b[clave]
        filas.append({"Clase": clave.capitalize(), "Modelo": modelo_a["nombre"], "Valor": modelo_a["por_clase"][clase_a].get(metrica)})
        filas.append({"Clase": clave.capitalize(), "Modelo": modelo_b["nombre"], "Valor": modelo_b["por_clase"][clase_b].get(metrica)})
    df = pd.DataFrame(filas)
    fig = px.bar(df, x="Clase", y="Valor", color="Modelo", barmode="group", range_y=[0, 1])
    fig.update_layout(
        title=f"{ETIQUETAS_METRICAS.get(metrica, metrica.replace('_', ' ').title())} por clase",
        margin=dict(t=40),
    )
    return aplicar_layout_responsivo(fig), None
