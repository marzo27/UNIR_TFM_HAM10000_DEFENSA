"""Carga de los resultados del EDA (Análisis Exploratorio de Datos) de HAM10000.

Fuentes reales (no se recalcula ni se inventa ningún valor):
    outputs/EDA/eda_resultados_HAM10000.xlsx  (12 hojas)
    outputs/EDA/eda_ham10000_v1_salida.txt    (log de ejecución; de aquí se
                                                extraen los dos tests
                                                estadísticos que no están en
                                                el Excel: Kruskal-Wallis y
                                                Point-Biserial)
    outputs/EDA/fig01..fig10_*.png            (figuras ya generadas por el script)
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# dashboard_gradio/utils/carga_eda.py -> dashboard_gradio/utils -> dashboard_gradio -> UNIR_TFM_HAM10000
EDA_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "EDA"
XLSX_PATH = EDA_DIR / "eda_resultados_HAM10000.xlsx"
TXT_PATH = EDA_DIR / "eda_ham10000_v1_salida.txt"

NOMBRES_HOJAS = {
    "resumen": "01_Resumen_General",
    "distribucion_clases": "02_Distribucion_Clases",
    "estadist_edad": "03_Estadist_Edad",
    "mediana_edad_imputac": "04_Mediana_Edad_Imputac",
    "distribucion_sexo": "05_Distribucion_Sexo",
    "distribucion_localizac": "06_Distribucion_Localizac",
    "distribucion_dxtype": "07_Distribucion_DxType",
    "crosstab_dx_dxtype": "08_CrossTab_Dx_DxType",
    "crosstab_loc_dx": "09_CrossTab_Loc_Dx",
    "cramers_v": "10_Cramers_V",
    "gruposedad_dx": "11_GruposEdad_Dx",
    "dataset_limpio": "12_Dataset_Limpio",
}


@lru_cache(maxsize=None)
def cargar_hoja(clave: str) -> pd.DataFrame:
    """Lee una hoja del Excel de EDA tal cual, identificada por una clave corta."""
    if clave not in NOMBRES_HOJAS:
        raise KeyError(f"Hoja de EDA desconocida: {clave!r}. Opciones: {list(NOMBRES_HOJAS)}")
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"No se encontró el Excel del EDA: {XLSX_PATH}")
    return pd.read_excel(XLSX_PATH, sheet_name=NOMBRES_HOJAS[clave])


@lru_cache(maxsize=1)
def cargar_resumen_dict() -> dict:
    """Convierte la hoja 01_Resumen_General en un diccionario {Métrica: Valor}."""
    df = cargar_hoja("resumen")
    return dict(zip(df["Métrica"], df["Valor"]))


@lru_cache(maxsize=1)
def cargar_tests_estadisticos() -> dict:
    """Extrae del log .txt los tests que no están en el Excel.

    Busca literalmente las líneas
    'Kruskal-Wallis (edad ~ dx): H = ..., p = ...' y
    'Point-Biserial (edad ~ malignidad): r = ..., p = ...'
    en el .txt exportado por el script de EDA. Si no las encuentra (p. ej.
    porque el archivo no está o cambió de formato), devuelve None en cada
    clave en vez de inventar un resultado.
    """
    resultado: dict = {"kruskal_wallis": None, "point_biserial": None}
    if not TXT_PATH.exists():
        return resultado
    texto = TXT_PATH.read_text(encoding="utf-8", errors="ignore")

    m = re.search(
        r"Kruskal-Wallis \(edad ~ dx\):\s*H\s*=\s*([\d.]+),\s*p\s*=\s*([\d.eE+-]+)", texto
    )
    if m:
        resultado["kruskal_wallis"] = {"H": float(m.group(1)), "p": float(m.group(2))}

    m = re.search(
        r"Point-Biserial \(edad ~ malignidad\):\s*r\s*=\s*([\d.]+),\s*p\s*=\s*([\d.eE+-]+)", texto
    )
    if m:
        resultado["point_biserial"] = {"r": float(m.group(1)), "p": float(m.group(2))}

    return resultado


FIGURAS = [
    ("fig01_distribucion_clases.png", "Distribución de clases (dx)"),
    ("fig02_distribucion_edad.png", "Distribución de edad"),
    ("fig03_distribucion_sexo.png", "Distribución de sexo"),
    ("fig04_localizacion.png", "Localización anatómica"),
    ("fig05_dx_type.png", "Tipo de diagnóstico (dx_type)"),
    ("fig06_malignidad_edad.png", "Malignidad vs. edad"),
    ("fig07_grupos_edad_dx.png", "Grupos de edad por clase dx"),
    ("fig08_correlacion_cramers_v.png", "Correlación entre variables (Cramér's V)"),
    ("fig09_imagenes_por_lesion.png", "Imágenes por lesión"),
    ("fig10_sexo_localizacion_apilado.png", "Sexo y localización (apilado)"),
]


def ruta_figura(nombre_archivo: str) -> Path:
    return EDA_DIR / nombre_archivo
