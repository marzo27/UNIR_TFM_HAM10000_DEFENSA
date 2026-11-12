# TFM — Evaluación comparativa de SVM y DenseNet201 para la clasificación automática de enfermedades dermatológicas
### Máster en Análisis y Visualización de Datos Masivos — UNIR

---

## Descripción

Este repositorio contiene el código fuente, resultados y documentación del **Trabajo de Fin de Máster (TFM)** del Máster Universitario en Análisis y Visualización de Datos Masivos de la Universidad Internacional de La Rioja (UNIR).

**Tema:** Clasificación automática de enfermedades dermatológicas mediante inteligencia artificial aplicada al dataset HAM10000.

---

## Dataset

Los datos utilizados en este proyecto son el dataset público **HAM10000 (Human Against Machine with 10,000 training images)**, disponible en Kaggle:

- **URL:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Autores:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018)
- **Licencia:** CC BY-NC-SA 4.0
- **Descripción:** 10.015 imágenes dermatoscópicas de 7 clases de lesiones cutáneas

> **Nota:** Por el tamaño de los archivos de imagen (>2 GB), los datos no están incluidos en este repositorio. Consulta la sección [Reproducción del entorno](#reproducción-del-entorno-de-datos) para obtener las instrucciones de descarga.

---

## Estructura del repositorio

```
UNIR_TFM_HAM10000/
├── data/
│   └── README_data.md
├── notebooks/
│   ├── TFM_1.ipynb
│   ├── TFM_1_v2.ipynb
│   ├── TFM_1_v3.ipynb
│   ├── TFM_1_v4.ipynb
│   ├── TFM_DenseNet201_Escenario1_Colab.ipynb
│   └── TFM_EDA_SVM.ipynb
├── outputs/
│   ├── DenseNet/
│   │   ├── Figure_1.png
│   │   ├── Figure_2.png
│   │   ├── Figure_3.png
│   │   ├── Figure_4.png
│   │   ├── Figure_5.png
│   │   ├── Figure_6.png
│   │   ├── Figure_7.png
│   │   ├── metricas_densenet201.json
│   │   └── Resultados.txt
│   ├── DenseNet_Balanceado/
│   │   ├── Figure_1.png
│   │   ├── Figure_2.png
│   │   ├── Figure_3.png
│   │   ├── Figure_4.png
│   │   ├── Figure_5.png
│   │   ├── Figure_6.png
│   │   ├── Figure_7.png
│   │   ├── metricas_densenet201.json
│   │   └── Resultados.txt
│   ├── EDA/
│   │   ├── eda_ham10000_v1_salida.txt
│   │   ├── eda_resultados_HAM10000.xlsx
│   │   ├── fig01_distribucion_clases.png
│   │   ├── fig02_distribucion_edad.png
│   │   ├── fig03_distribucion_sexo.png
│   │   ├── fig04_localizacion.png
│   │   ├── fig05_dx_type.png
│   │   ├── fig06_malignidad_edad.png
│   │   ├── fig07_grupos_edad_dx.png
│   │   ├── fig08_correlacion_cramers_v.png
│   │   ├── fig09_imagenes_por_lesion.png
│   │   └── fig10_sexo_localizacion_apilado.png
│   ├── SVM/
│   │   ├── plots_svm/
│   │   │   ├── 01_distribucion_clases.png
│   │   │   ├── 02_analisis_pca.png
│   │   │   ├── 03_matriz_confusion.png
│   │   │   ├── 04_curvas_roc.png
│   │   │   ├── 05_precision_recall.png
│   │   │   ├── 06_metricas_por_clase.png
│   │   │   ├── 07_validacion_cruzada.png
│   │   │   └── 08_radar_metricas.png
│   │   ├── metricas.xlsx
│   │   ├── metricas_baseline.json
│   │   ├── svm_ham10000_v1_salida.txt
│   │   └── svm_model.joblib
│   └── SVM_Balanceado/
│       ├── plots_svm_balanceado/
│       │   ├── 01_distribucion_clases.png
│       │   ├── 02_analisis_pca.png
│       │   ├── 03_matriz_confusion.png
│       │   ├── 04_curvas_roc.png
│       │   ├── 05_precision_recall.png
│       │   ├── 06_metricas_por_clase.png
│       │   ├── 07_validacion_cruzada.png
│       │   └── 08_radar_metricas.png
│       ├── metricas_balanceado.json
│       ├── metricas_balanceado.xlsx
│       ├── svm_ham10000_balanceado_salida.txt
│       └── svm_model_balanceado.joblib
├── src/
│   ├── densenet201_ham10000.py
│   ├── densenet201_ham10000_balanceado.py
│   ├── eda_ham10000.py
│   ├── eda_ham10000_v1.py
│   ├── generar_dataset_balanceado.py
│   ├── svm_ham10000.py
│   ├── svm_ham10000_balanceado.py
│   └── svm_ham10000_v1.py
├── actualizar_estructura_readme.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Reproducción del entorno de datos

### Paso 1 — Instalar la Kaggle API

```bash
pip install kaggle
```

### Paso 2 — Configurar credenciales de Kaggle

1. Inicia sesión en https://www.kaggle.com
2. Ve a **Account → API → Create New Token**
3. Descarga el archivo `kaggle.json`
4. Colócalo en:
   - Windows: `C:\Users\TU_USUARIO\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

### Paso 3 — Descargar el dataset

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/
```

### Paso 4 — Descomprimir

```bash
cd data
unzip skin-cancer-mnist-ham10000.zip -d HAM10000
```

La estructura resultante debe quedar así:

```
data/
└── HAM10000/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    ├── HAM10000_images_part_2/
    ├── hmnist_28_28_L.csv
    ├── hmnist_28_28_RGB.csv
    ├── hmnist_8_8_L.csv
    └── hmnist_8_8_RGB.csv
```

---

## Instalación del entorno Python

### Opción A — Entorno virtual (recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción B — Instalación directa

```bash
pip install -r requirements.txt
```

---

## Ejecución del EDA

Antes de ejecutar, ajusta las rutas en el script si es necesario. Por defecto el script busca los datos en `data/HAM10000/`:

```bash
python src/eda_ham10000_v1.py
```

Los resultados se guardan automáticamente en `outputs/`.

---

## Resultados del EDA

El script genera:

- **10 figuras PNG** en `outputs/figures/` con análisis de distribución de clases, edad, sexo, localización, método diagnóstico y correlaciones
- **1 archivo Excel** en `outputs/excel/` con 12 hojas de estadísticas detalladas
- **Salida en consola** con todos los estadísticos descriptivos y resultados de tests

---

## Referencia del dataset

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, *5*, 180161. https://doi.org/10.1038/sdata.2018.161

---

## Licencia del código

El código fuente de este repositorio está bajo licencia **MIT**. El dataset HAM10000 está sujeto a su propia licencia (CC BY-NC-SA 4.0); consulta las condiciones de uso en Kaggle antes de utilizarlo.
