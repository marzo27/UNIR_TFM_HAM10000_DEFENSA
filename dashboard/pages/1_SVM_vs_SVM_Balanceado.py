import sys
from pathlib import Path

import streamlit as st

# Streamlit no añade automáticamente la raíz de la app al sys.path al
# ejecutar páginas dentro de pages/, así que lo hacemos explícito aquí
# para poder importar el paquete utils/ compartido entre las 3 páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.carga_metricas import cargar_svm
from utils.render import (
    galeria_imagenes,
    grafico_barras_comparativo,
    kpis_resumen,
    tabla_comparativa_global,
    tabla_por_clase,
)

st.set_page_config(page_title="SVM vs SVM Balanceado", layout="wide", page_icon="⚔️")
st.title("⚔️ SVM vs SVM Balanceado")

svm = cargar_svm("SVM")
svm_bal = cargar_svm("SVM_Balanceado")

st.warning(
    f"**Nota de comparabilidad:** SVM (baseline) resuelve un problema multiclase "
    f"de {svm['n_clases']} clases ({', '.join(svm['clases'])}), mientras que "
    f"SVM_Balanceado resuelve un problema **binario** ({', '.join(svm_bal['clases'])}) "
    f"sobre un conjunto rebalanceado. Las métricas globales son comparables porque "
    f"usan la misma fórmula; las tablas por clase se muestran por separado porque "
    f"las clases no coinciden."
)

st.header("Resumen")
kpis_resumen(svm, svm_bal)

st.header("Métricas globales")
c1, c2 = st.columns([1, 1.3])
with c1:
    st.dataframe(tabla_comparativa_global(svm, svm_bal), width='stretch')
with c2:
    grafico_barras_comparativo(svm, svm_bal)

st.header("Métricas por clase")
c1, c2 = st.columns(2)
with c1:
    st.subheader(svm["nombre"])
    st.dataframe(tabla_por_clase(svm), width='stretch')
with c2:
    st.subheader(svm_bal["nombre"])
    st.dataframe(tabla_por_clase(svm_bal), width='stretch')

st.header("Validación cruzada")
c1, c2 = st.columns(2)
for col, modelo in zip([c1, c2], [svm, svm_bal]):
    with col:
        cv = modelo.get("cv") or {}
        st.metric(
            f"{modelo['nombre']} — Accuracy CV ({cv.get('n_folds', '?')} folds)",
            f"{cv.get('cv_acc_mean', float('nan')):.4f}",
            f"± {cv.get('cv_acc_std', float('nan')):.4f}",
        )

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
tab_a, tab_b = st.tabs([svm["nombre"], svm_bal["nombre"]])
with tab_a:
    galeria_imagenes("", [(svm["plots_dir"] / f, c) for f, c in nombres_svm])
with tab_b:
    galeria_imagenes("", [(svm_bal["plots_dir"] / f, c) for f, c in nombres_svm])

with st.expander("Ver JSON crudo"):
    c1, c2 = st.columns(2)
    c1.json(svm["raw"])
    c2.json(svm_bal["raw"])
