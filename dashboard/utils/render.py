"""Componentes de UI reutilizables por las páginas de comparación."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
    filas = {}
    for clave, etiqueta in ETIQUETAS_METRICAS.items():
        val_a = modelo_a["global"].get(clave)
        val_b = modelo_b["global"].get(clave)
        if val_a is None and val_b is None:
            continue
        filas[etiqueta] = {modelo_a["nombre"]: val_a, modelo_b["nombre"]: val_b}
    df = pd.DataFrame(filas).T
    return df.round(4)


def kpis_resumen(modelo_a: dict, modelo_b: dict) -> None:
    cols = st.columns(len(METRICAS_KPI))
    for col, clave in zip(cols, METRICAS_KPI):
        val_a = modelo_a["global"].get(clave)
        val_b = modelo_b["global"].get(clave)
        if val_a is None or val_b is None:
            col.metric(ETIQUETAS_METRICAS.get(clave, clave), "N/D")
            continue
        delta = val_b - val_a
        col.metric(
            ETIQUETAS_METRICAS.get(clave, clave),
            f"{val_b:.4f}",
            f"{delta:+.4f} vs {modelo_a['nombre']}",
        )


def grafico_barras_comparativo(modelo_a: dict, modelo_b: dict) -> None:
    df = tabla_comparativa_global(modelo_a, modelo_b).reset_index().rename(columns={"index": "Métrica"})
    df_plot = df.melt(id_vars="Métrica", var_name="Modelo", value_name="Valor")
    fig = px.bar(
        df_plot, x="Métrica", y="Valor", color="Modelo", barmode="group",
        range_y=[0, 1], text_auto=".3f",
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')


def tabla_por_clase(modelo: dict) -> pd.DataFrame:
    df = pd.DataFrame(modelo["por_clase"]).T
    return df.round(4)


def graficos_por_clase_comparados(modelo_a: dict, modelo_b: dict, metrica: str = "f1_score") -> None:
    """Compara una métrica por clase entre dos modelos.

    Empareja clases ignorando mayúsculas/minúsculas (p. ej. "Benigno" del SVM
    con "benigno" de DenseNet) sin alterar los nombres originales mostrados.
    Si los modelos no comparten ninguna clase, no dibuja nada y avisa.
    """
    mapa_a = {c.lower(): c for c in modelo_a["clases"]}
    mapa_b = {c.lower(): c for c in modelo_b["clases"]}
    comunes = [k for k in mapa_a if k in mapa_b]
    if not comunes:
        st.warning(
            "Los modelos no comparten clases directamente comparables "
            f"({modelo_a['nombre']}: {modelo_a['clases']} vs "
            f"{modelo_b['nombre']}: {modelo_b['clases']})."
        )
        return

    filas = []
    for clave in comunes:
        clase_a, clase_b = mapa_a[clave], mapa_b[clave]
        filas.append({"Clase": clave.capitalize(), "Modelo": modelo_a["nombre"], "Valor": modelo_a["por_clase"][clase_a].get(metrica)})
        filas.append({"Clase": clave.capitalize(), "Modelo": modelo_b["nombre"], "Valor": modelo_b["por_clase"][clase_b].get(metrica)})
    df = pd.DataFrame(filas)
    fig = px.bar(df, x="Clase", y="Valor", color="Modelo", barmode="group", range_y=[0, 1])
    fig.update_layout(title=f"{ETIQUETAS_METRICAS.get(metrica, metrica.replace('_', ' ').title())} por clase")
    st.plotly_chart(fig, width='stretch')


def galeria_imagenes(titulo: str, rutas_con_captions: list[tuple[Path, str]]) -> None:
    if titulo:
        st.subheader(titulo)
    cols = st.columns(2)
    for i, (ruta, caption) in enumerate(rutas_con_captions):
        with cols[i % 2]:
            if ruta.exists():
                st.image(str(ruta), caption=caption, width='stretch')
            else:
                st.info(f"No se encontró la imagen: {ruta}")
