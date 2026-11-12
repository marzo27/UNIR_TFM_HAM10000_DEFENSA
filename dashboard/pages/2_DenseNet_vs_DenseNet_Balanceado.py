import sys
from pathlib import Path

import streamlit as st

# Streamlit no añade automáticamente la raíz de la app al sys.path al
# ejecutar páginas dentro de pages/, así que lo hacemos explícito aquí
# para poder importar el paquete utils/ compartido entre las 3 páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.carga_metricas import cargar_densenet
from utils.render import (
    galeria_imagenes,
    grafico_barras_comparativo,
    graficos_por_clase_comparados,
    kpis_resumen,
    tabla_comparativa_global,
    tabla_por_clase,
)

st.set_page_config(page_title="DenseNet vs DenseNet Balanceado", layout="wide", page_icon="⚔️")
st.title("⚔️ DenseNet201 vs DenseNet201 Balanceado")

cabeza = st.radio(
    "Cabeza del modelo a comparar",
    options=["7_clases", "binaria"],
    format_func=lambda x: (
        "Diagnóstico específico (7 clases)" if x == "7_clases" else "Triaje binario (Benigno / Maligno)"
    ),
    horizontal=True,
)

dn = cargar_densenet("DenseNet", cabeza=cabeza)
dn_bal = cargar_densenet("DenseNet_Balanceado", cabeza=cabeza)

st.caption(
    "DenseNet201 es multitarea: exporta una cabeza de diagnóstico específico (7 clases) "
    "y una cabeza de triaje binario en el mismo entrenamiento. Ambas versiones (original "
    f"y balanceada) comparten exactamente las mismas clases para la cabeza seleccionada "
    f"({', '.join(dn['clases'])}), por lo que la comparación por clase es directa."
)

st.header("Resumen")
kpis_resumen(dn, dn_bal)

st.header("Métricas globales")
c1, c2 = st.columns([1, 1.3])
with c1:
    st.dataframe(tabla_comparativa_global(dn, dn_bal), width='stretch')
with c2:
    grafico_barras_comparativo(dn, dn_bal)

st.header("Métricas por clase")
graficos_por_clase_comparados(dn, dn_bal, metrica="f1_score")
c1, c2 = st.columns(2)
with c1:
    st.subheader(dn["nombre"])
    st.dataframe(tabla_por_clase(dn), width='stretch')
with c2:
    st.subheader(dn_bal["nombre"])
    st.dataframe(tabla_por_clase(dn_bal), width='stretch')

st.header("Gráficos generados durante el entrenamiento")
st.caption(
    "Orden de figuras inferido de `src/densenet201_ham10000*.py`, a partir de los "
    "títulos `fig.suptitle(...)` en el mismo orden en que se generan (Figure_1.png es "
    "el primer gráfico creado, Figure_2.png el segundo, etc.)."
)
nombres_figuras = [
    ("Figure_1.png", "Matriz de confusión — Cabeza B (7 clases)"),
    ("Figure_2.png", "Matriz de confusión — Cabeza A (binaria)"),
    ("Figure_3.png", "Curvas ROC — Cabeza B (One-vs-Rest)"),
    ("Figure_4.png", "Curvas Precision-Recall — Cabeza B"),
    ("Figure_5.png", "ROC + Precision-Recall — Cabeza A (binaria)"),
    ("Figure_6.png", "Métricas por clase — Cabeza B"),
    ("Figure_7.png", "Curvas de entrenamiento (loss / accuracy)"),
]
tab_a, tab_b = st.tabs(["DenseNet", "DenseNet Balanceado"])
with tab_a:
    galeria_imagenes("", [(dn["plots_dir"] / f, c) for f, c in nombres_figuras])
with tab_b:
    galeria_imagenes("", [(dn_bal["plots_dir"] / f, c) for f, c in nombres_figuras])

with st.expander("Ver JSON crudo"):
    c1, c2 = st.columns(2)
    c1.json(dn["raw"])
    c2.json(dn_bal["raw"])
