"""Localización de las imágenes originales de HAM10000.

Las imágenes NO viven dentro del repositorio del proyecto
(UNIR_TFM_HAM10000), sino en una carpeta de datos aparte, hermana de
UNIR_TFM_HAM10000 bajo D:\\TFM:

    D:\\TFM\\HAM10000\\HAM10000_images_part_1\\<image_id>.jpg
    D:\\TFM\\HAM10000\\HAM10000_images_part_2\\<image_id>.jpg

(confirmado: 5000 + 5015 = 10015 imágenes .jpg, una por cada fila de
`image_id` del dataset).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# dashboard_gradio/utils/carga_imagenes.py -> utils -> dashboard_gradio -> UNIR_TFM_HAM10000 -> TFM
TFM_DIR = Path(__file__).resolve().parent.parent.parent.parent
IMAGENES_DIRS = [
    TFM_DIR / "HAM10000" / "HAM10000_images_part_1",
    TFM_DIR / "HAM10000" / "HAM10000_images_part_2",
]


@lru_cache(maxsize=1)
def _indice_imagenes() -> dict:
    """Escanea una sola vez ambas carpetas y mapea image_id -> ruta completa."""
    indice: dict[str, Path] = {}
    for carpeta in IMAGENES_DIRS:
        if not carpeta.exists():
            continue
        for archivo in carpeta.glob("*.jpg"):
            indice[archivo.stem] = archivo
    return indice


def carpetas_imagenes_disponibles() -> bool:
    return any(c.exists() for c in IMAGENES_DIRS)


def buscar_imagen(image_id: str) -> Path | None:
    """Devuelve la ruta completa de la imagen para un image_id, o None si no está."""
    return _indice_imagenes().get(image_id)
