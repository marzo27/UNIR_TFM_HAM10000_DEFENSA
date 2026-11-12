"""Figuras de Plotly para la pestaña EDA, construidas a partir de las hojas
del Excel de EDA (ver utils.carga_eda). Funciones puras, sin dependencia de
ningún framework de UI; no recalculan nada que no esté ya en el Excel."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

from utils.plot_utils import aplicar_layout_responsivo


def fig_distribucion_clases(df_clases: pd.DataFrame):
    df = df_clases.sort_values("n_imagenes", ascending=False)
    fig = px.bar(
        df, x="dx", y="n_imagenes", color="malignidad", text="porcentaje_%",
        labels={"dx": "Clase (dx)", "n_imagenes": "N.º de imágenes", "malignidad": "Malignidad"},
        title="Distribución de imágenes por clase diagnóstica",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    return aplicar_layout_responsivo(fig)


def fig_malignidad_pie(df_clases: pd.DataFrame):
    resumen = df_clases.groupby("malignidad", as_index=False)[["n_imagenes", "porcentaje_%"]].sum()
    fig = px.pie(resumen, names="malignidad", values="n_imagenes", title="Benigno vs. Maligno (global)", hole=0.45)
    return aplicar_layout_responsivo(fig)


def fig_mediana_edad(df_mediana: pd.DataFrame):
    df = df_mediana.sort_values("mediana_edad_imputada", ascending=False)
    fig = px.bar(
        df, x="dx", y="mediana_edad_imputada", text="mediana_edad_imputada", hover_data=["nombre_largo"],
        labels={"dx": "Clase (dx)", "mediana_edad_imputada": "Mediana de edad (años)"},
        title="Mediana de edad usada para imputar valores nulos",
    )
    return aplicar_layout_responsivo(fig)


def fig_sexo_pie(df_sexo: pd.DataFrame):
    df = df_sexo.fillna({"sexo": "Sin dato"})
    fig = px.pie(df, names="sexo", values="conteo", title="Distribución por sexo", hole=0.45)
    return aplicar_layout_responsivo(fig)


def fig_localizacion_barh(df_loc: pd.DataFrame):
    df = df_loc.fillna({"localizacion": "Sin dato"}).sort_values("conteo", ascending=True)
    fig = px.bar(
        df, x="conteo", y="localizacion", orientation="h", text="porcentaje_%",
        labels={"conteo": "N.º de imágenes", "localizacion": "Localización"},
        title="Distribución por localización anatómica",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    return aplicar_layout_responsivo(fig)


def fig_heatmap_loc_dx(df_crosstab_loc: pd.DataFrame):
    matriz = df_crosstab_loc.set_index("localization")
    fig = px.imshow(
        matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Clase (dx)", y="Localización", color="N.º imágenes"),
        title="Localización × clase diagnóstica (dx)",
    )
    return aplicar_layout_responsivo(fig, height=500)


def fig_dxtype_bar(df_dxtype: pd.DataFrame):
    df = df_dxtype.sort_values("conteo", ascending=False)
    fig = px.bar(
        df, x="dx_type", y="conteo", text="porcentaje_%", hover_data=["nombre_largo"],
        labels={"dx_type": "Tipo de diagnóstico", "conteo": "N.º de imágenes"},
        title="Método de confirmación diagnóstica",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    return aplicar_layout_responsivo(fig)


def fig_heatmap_dx_dxtype(df_crosstab_dxtype: pd.DataFrame):
    matriz = df_crosstab_dxtype.set_index("dx")
    fig = px.imshow(
        matriz, text_auto=True, color_continuous_scale="Purples", aspect="auto",
        labels=dict(x="Tipo de diagnóstico", y="Clase (dx)", color="N.º imágenes"),
        title="Clase (dx) × tipo de diagnóstico",
    )
    return aplicar_layout_responsivo(fig)


def fig_cramers_v(df_cramer: pd.DataFrame):
    df = df_cramer.rename(columns={"Unnamed: 0": "variable"}).set_index("variable")
    fig = px.imshow(
        df, text_auto=".3f", color_continuous_scale="RdPu", zmin=0, zmax=1,
        labels=dict(color="Cramér's V"),
        title="Matriz de correlación — Cramér's V (variables categóricas)",
    )
    return aplicar_layout_responsivo(fig, height=480)


def fig_grupos_edad(df_grupos: pd.DataFrame):
    df = df_grupos.set_index("age_group")
    fig = px.imshow(
        df, text_auto=True, color_continuous_scale="Oranges", aspect="auto",
        labels=dict(x="Clase (dx)", y="Grupo de edad", color="N.º imágenes"),
        title="Grupo de edad × clase diagnóstica (dx)",
    )
    return aplicar_layout_responsivo(fig, height=450)
