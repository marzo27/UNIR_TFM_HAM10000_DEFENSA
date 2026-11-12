import streamlit as st

st.set_page_config(page_title="Comparación de modelos — HAM10000", layout="wide", page_icon="🔬")

st.title("🔬 Comparación de modelos — HAM10000")

st.markdown(
    """
    Panel interactivo para comparar el rendimiento de los modelos entrenados
    sobre el dataset **HAM10000** (clasificación de lesiones cutáneas).

    Usa el menú de la izquierda para navegar entre las cuatro secciones:

    0. **EDA** — análisis exploratorio del dataset original
    1. **SVM vs SVM Balanceado**
    2. **DenseNet201 vs DenseNet201 Balanceado**
    3. **SVM Balanceado vs DenseNet201 Balanceado**

    """
)

 #   Los datos se leen directamente de los archivos generados por los scripts
 #   del proyecto (`outputs/EDA`, `outputs/SVM`, `outputs/SVM_Balanceado`,
 #   `outputs/DenseNet`, `outputs/DenseNet_Balanceado`). No se recalculan ni
 #   se inventan métricas: se muestran tal como fueron exportadas.

# st.info(
#    "Esta app espera vivir en `UNIR_TFM_HAM10000/dashboard/` para localizar "
#    "automáticamente la carpeta `outputs/` (ruta relativa, sin depender de la unidad D:)."
#)

with st.expander("Notas importantes sobre comparabilidad"):
    st.markdown(
        """
        - **SVM (baseline)** resuelve un problema **multiclase de 7 clases**
          (akiec, bcc, bkl, df, mel, nv, vasc). **SVM_Balanceado** resuelve un
          problema **binario** (Benigno/Maligno) sobre un conjunto rebalanceado:
          no son la misma tarea, así que sus tablas por clase no se cruzan.
        - **DenseNet201** es un modelo **multitarea**: exporta a la vez una
          cabeza de diagnóstico específico (7 clases) y una cabeza de triaje
          binario (Benigno/Maligno). Ambas versiones (original y balanceada)
          comparten exactamente las mismas clases, así que su comparación por
          clase sí es directa en las dos cabezas.
        - Para comparar **SVM_Balanceado vs DenseNet201_Balanceado** de forma
          justa se usa por defecto la cabeza **binaria** de DenseNet, ya que
          es la que coincide con la tarea del SVM balanceado.
        """
    )
