"""Ajustes de layout compartidos para que las figuras de Plotly se vean bien
dentro de gr.Plot.

Sin esto, las figuras de plotly.express no llevan una altura ni un modo de
autosize explícitos; Gradio las monta con el tamaño que le da el contenedor
en ese instante, y si no coincide con el tamaño "natural" de la figura,
Plotly la reescala de forma desproporcionada (efecto de zoom/recorte). Fijar
una altura explícita + autosize=True evita esa dependencia del contenedor.

Además se fijan márgenes reducidos: los márgenes por defecto de Plotly son
grandes (pensados para dar sitio a títulos/leyendas independientemente del
tamaño del gráfico), así que al fijar una altura explícita ese margen pasa a
ocupar muchos más píxeles reales, dejando bastante espacio en blanco por
encima y por debajo del área dibujada.
"""
from __future__ import annotations

import plotly.graph_objects as go

ALTURA_DEFECTO = 420
MARGEN_DEFECTO = dict(l=50, r=30, t=50, b=40)


def aplicar_layout_responsivo(
    fig: go.Figure,
    height: int = ALTURA_DEFECTO,
    margin: dict | None = None,
) -> go.Figure:
    fig.update_layout(autosize=True, height=height, margin=margin or MARGEN_DEFECTO)
    return fig
