"""Carga y normalización de las métricas exportadas por los scripts de entrenamiento.

Los JSON generados por los scripts SVM (`src/svm_ham10000*.py`) y DenseNet
(`src/densenet201_ham10000*.py`) tienen esquemas distintos entre sí. Este
módulo los lee TAL CUAL fueron exportados y los homogeneiza a una estructura
común únicamente para poder mostrarlos lado a lado; no se recalcula ni se
inventa ningún valor — cada campo normalizado documenta de qué clave del
JSON original proviene.

Archivos fuente reales:
    outputs/SVM/metricas_baseline.json
    outputs/SVM_Balanceado/metricas_balanceado.json
    outputs/DenseNet/metricas_densenet201.json
    outputs/DenseNet_Balanceado/metricas_densenet201.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

# dashboard/utils/carga_metricas.py -> dashboard/utils -> dashboard -> UNIR_TFM_HAM10000
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def _leer_json(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de métricas esperado: {ruta}\n"
            f"Comprueba que la carpeta 'dashboard' esté dentro de UNIR_TFM_HAM10000."
        )
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SVM (baseline / balanceado) — esquema con "global_metrics" / "per_class_metrics"
# ---------------------------------------------------------------------------

CarpetaSVM = Literal["SVM", "SVM_Balanceado"]


def cargar_svm(carpeta: CarpetaSVM) -> dict:
    """Carga y normaliza las métricas de un modelo SVM (baseline o balanceado).

    Mapeo real (sin inventar valores):
        global    <- data["global_metrics"]
        por_clase <- data["per_class_metrics"]  (avg_prec se expone como avg_precision)
    """
    nombre_archivo = "metricas_baseline.json" if carpeta == "SVM" else "metricas_balanceado.json"
    data = _leer_json(OUTPUTS_DIR / carpeta / nombre_archivo)

    por_clase = {
        clase: {
            "precision": v["precision"],
            "recall": v["recall"],
            "f1_score": v["f1_score"],
            "support": v["support"],
            "auc": v.get("auc"),
            "avg_precision": v.get("avg_prec"),
        }
        for clase, v in data["per_class_metrics"].items()
    }

    carpeta_plots = "plots_svm" if carpeta == "SVM" else "plots_svm_balanceado"

    return {
        "nombre": data.get("model_name", carpeta),
        "carpeta": carpeta,
        "clases": data["classes"],
        "n_clases": data["n_classes"],
        "n_train": data.get("n_train"),
        "n_test": data.get("n_test"),
        "global": dict(data["global_metrics"]),
        "por_clase": por_clase,
        "cv": data.get("cross_validation"),
        "tiempos": data.get("timing_seconds"),
        "timestamp": data.get("timestamp"),
        "plots_dir": OUTPUTS_DIR / carpeta / carpeta_plots,
        "raw": data,
    }


# ---------------------------------------------------------------------------
# DenseNet201 (original / balanceado) — modelo multitarea con dos cabezas
# ---------------------------------------------------------------------------

CarpetaDenseNet = Literal["DenseNet", "DenseNet_Balanceado"]
CabezaDenseNet = Literal["7_clases", "binaria"]


def cargar_densenet(carpeta: CarpetaDenseNet, cabeza: CabezaDenseNet = "7_clases") -> dict:
    """Carga y normaliza las métricas de un modelo DenseNet201 (multitarea).

    DenseNet201 exporta dos cabezas en el mismo JSON:
        cabeza_b_7_clases: diagnóstico específico (7 clases de lesión)
        cabeza_a_binaria : triaje binario (Benigno / Maligno)

    Mapeo real (sin inventar valores). La cabeza de 7 clases usa las mismas
    claves que el esquema de SVM (precision_macro, recall_macro, auc_macro...).
    La cabeza binaria NO reporta esas claves; en su lugar usa
    'precision_maligno', 'sensibilidad_maligno' (= recall de la clase
    Maligno) y 'auc_roc'. Aquí se exponen bajo el mismo nombre normalizado
    para poder comparar, y además se conservan sin tocar en 'global_original'.
    """
    data = _leer_json(OUTPUTS_DIR / carpeta / "metricas_densenet201.json")

    if cabeza == "7_clases":
        bloque = data["cabeza_b_7_clases"]
        clases = data["clases"]
        global_norm = {
            "accuracy": bloque.get("accuracy"),
            "balanced_accuracy": bloque.get("balanced_accuracy"),
            "f1_macro": bloque.get("f1_macro"),
            "f1_weighted": bloque.get("f1_weighted"),
            "precision_macro": bloque.get("precision_macro"),
            "recall_macro": bloque.get("recall_macro"),
            "auc_macro": bloque.get("auc_macro"),
            "cohen_kappa": bloque.get("cohen_kappa"),
            "mcc": bloque.get("mcc"),
        }
        por_clase = {
            clase: {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1_score": v["f1_score"],
                "support": v["support"],
                "auc": v.get("auc"),
                "avg_precision": v.get("avg_precision"),
            }
            for clase, v in bloque["por_clase"].items()
        }
    else:  # binaria
        bloque = data["cabeza_a_binaria"]
        clases = list(bloque["por_clase"].keys())  # ["benigno", "maligno"], tal cual el JSON
        global_norm = {
            "accuracy": bloque.get("accuracy"),
            "balanced_accuracy": bloque.get("balanced_accuracy"),
            "f1_macro": bloque.get("f1_macro"),
            "f1_weighted": bloque.get("f1_weighted"),
            "precision_macro": bloque.get("precision_maligno"),  # ver nota en docstring
            "recall_macro": bloque.get("sensibilidad_maligno"),  # ver nota en docstring
            "auc_macro": bloque.get("auc_roc"),  # ver nota en docstring
            "cohen_kappa": bloque.get("cohen_kappa"),
            "mcc": bloque.get("mcc"),
        }
        por_clase = {
            clase: {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1_score": v["f1_score"],
                "support": v["support"],
                "auc": None,  # no reportado por clase para la cabeza binaria
                "avg_precision": None,
            }
            for clase, v in bloque["por_clase"].items()
        }

    etiqueta_cabeza = "7 clases" if cabeza == "7_clases" else "binaria"
    # OJO: el campo "modelo" del JSON es literalmente "DenseNet201_Multitask"
    # en AMBAS carpetas (DenseNet y DenseNet_Balanceado) — usarlo tal cual
    # produce el mismo "nombre" para los dos modelos y rompe cualquier tabla
    # que use el nombre como clave de columna (una versión pisa a la otra).
    # Por eso el nombre se construye a partir de la carpeta, que sí es única.
    etiqueta_carpeta = "DenseNet" if carpeta == "DenseNet" else "DenseNet Balanceado"

    return {
        "nombre": f"{etiqueta_carpeta} ({etiqueta_cabeza})",
        "carpeta": carpeta,
        "cabeza": cabeza,
        "clases": clases,
        "n_clases": len(clases),
        "n_train": data.get("n_train"),
        "n_test": data.get("n_val"),
        "global": global_norm,
        "global_original": dict(bloque),
        "por_clase": por_clase,
        "tiempos": data.get("tiempos_segundos"),
        "timestamp": data.get("timestamp"),
        "plots_dir": OUTPUTS_DIR / carpeta,
        "raw": data,
    }
