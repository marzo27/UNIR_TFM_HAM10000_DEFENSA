import sys
from pathlib import Path

import streamlit as st

# Streamlit no añade automáticamente la raíz de la app al sys.path al
# ejecutar páginas dentro de pages/, así que lo hacemos explícito aquí
# para poder importar el paquete utils/ compartido entre las 3 páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.carga_metricas import cargar_densenet, cargar_svm
from utils.render import (
    galeria_imagenes,
    grafico_barras_comparativo,
    graficos_por_clase_comparados,
    kpis_resumen,
    tabla_comparativa_global,
    tabla_por_clase,
)

st.set_page_config(page_title="SVM Balanceado vs DenseNet Balanceado", layout="wide", page_icon="⚔️")
st.title("⚔️ SVM Balanceado vs DenseNet201 Balanceado")

cabeza = st.radio(
    "Cabeza de DenseNet a comparar contra el SVM Balanceado (binario)",
    options=["binaria", "7_clases"],
    format_func=lambda x: (
        "Triaje binario (Benigno / Maligno) — comparable 1:1"
        if x == "binaria"
        else "Diagnóstico específico (7 clases) — no comparable por clase"
    ),
    horizontal=True,
)

svm_bal = cargar_svm("SVM_Balanceado")
dn_bal = cargar_densenet("DenseNet_Balanceado", cabeza=cabeza)

if cabeza == "binaria":
    st.info(
        "Ambos modelos resuelven el mismo problema binario (Benigno vs. Maligno) "
        "sobre el conjunto balanceado, por lo que la comparación es directa."
    )
else:
    st.warning(
        "SVM_Balanceado es un clasificador **binario** (Benigno/Maligno), mientras que "
        "aquí se usa la cabeza de **7 clases** de DenseNet. Las métricas globales siguen "
        "siendo comparables, pero la tabla por clase no lo es (2 clases vs 7)."
    )

st.header("Resumen")
kpis_resumen(svm_bal, dn_bal)

st.header("Métricas globales")
c1, c2 = st.columns([1, 1.3])
with c1:
    st.dataframe(tabla_comparativa_global(svm_bal, dn_bal), width='stretch')
with c2:
    grafico_barras_comparativo(svm_bal, dn_bal)

st.header("Métricas por clase")
if cabeza == "binaria":
    graficos_por_clase_comparados(svm_bal, dn_bal, metrica="f1_score")
c1, c2 = st.columns(2)
with c1:
    st.subheader(svm_bal["nombre"])
    st.dataframe(tabla_por_clase(svm_bal), width='stretch')
with c2:
    st.subheader(dn_bal["nombre"])
    st.dataframe(tabla_por_clase(dn_bal), width='stretch')

st.header("Gráficos generados durante el entrenamiento")
nombres_svm = [
    ("01_distribucion_clases.png", "Distribución de clases"),
    ("02_analisis_pca.png", "Análisis PCA"),
    ("03_matriz_confusion.png", "Matriz de confusión"),
    ("04_curvas_roc.png", "Curvas ROC"),
    ("05_precision_recall.png", "Precision-Recall"),
    ("06_metricas_por_clase.png", "Métricas por clase"),
    ("07_validacion_cruzada.png", "Validación cruzada"),
    ("08_radar_metricas.png", "Radar de métricas"),
]
nombres_figuras_dn = [
    ("Figure_1.png", "Matriz de confusión — Cabeza B (7 clases)"),
    ("Figure_2.png", "Matriz de confusión — Cabeza A (binaria)"),
    ("Figure_3.png", "Curvas ROC — Cabeza B (One-vs-Rest)"),
    ("Figure_4.png", "Curvas Precision-Recall — Cabeza B"),
    ("Figure_5.png", "ROC + Precision-Recall — Cabeza A (binaria)"),
    ("Figure_6.png", "Métricas por clase — Cabeza B"),
    ("Figure_7.png", "Curvas de entrenamiento (loss / accuracy)"),
]
tab_a, tab_b = st.tabs(["SVM Balanceado", "DenseNet Balanceado"])
with tab_a:
    galeria_imagenes("", [(svm_bal["plots_dir"] / f, c) for f, c in nombres_svm])
with tab_b:
    galeria_imagenes("", [(dn_bal["plots_dir"] / f, c) for f, c in nombres_figuras_dn])

with st.expander("Ver JSON crudo"):
    c1, c2 = st.columns(2)
    c1.json(svm_bal["raw"])
    c2.json(dn_bal["raw"])
