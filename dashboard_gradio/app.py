"""Dashboard comparativo HAM10000 — versión Gradio.

Mismas 4 secciones que la versión Streamlit (carpeta `dashboard/`), pero
construidas con gr.Blocks:

    0. EDA
    1. SVM vs SVM Balanceado
    2. DenseNet201 vs DenseNet201 Balanceado
    3. SVM Balanceado vs DenseNet201 Balanceado

Reutiliza exactamente los mismos módulos de carga de datos
(utils/carga_metricas.py, utils/carga_eda.py, utils/carga_imagenes.py) que la
versión Streamlit: no se recalcula ni se inventa ninguna métrica, todo se lee
de los archivos ya generados en outputs/.

Ejecutar con:
    python app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.carga_eda import (
    FIGURAS as FIGURAS_EDA,
    cargar_hoja,
    cargar_resumen_dict,
    cargar_tests_estadisticos,
    ruta_figura,
)
from utils.carga_imagenes import IMAGENES_DIRS, buscar_imagen, carpetas_imagenes_disponibles
from utils.carga_metricas import cargar_densenet, cargar_svm
from utils.comparaciones import (
    grafico_barras_comparativo,
    graficos_por_clase_comparados,
    tabla_comparativa_global,
    tabla_kpis_delta,
    tabla_por_clase,
)
from utils.graficos_eda import (
    fig_cramers_v,
    fig_distribucion_clases,
    fig_dxtype_bar,
    fig_grupos_edad,
    fig_heatmap_dx_dxtype,
    fig_heatmap_loc_dx,
    fig_localizacion_barh,
    fig_malignidad_pie,
    fig_mediana_edad,
    fig_sexo_pie,
)

NOMBRES_SVM_PLOTS = [
    ("01_distribucion_clases.png", "Distribución de clases"),
    ("02_analisis_pca.png", "Análisis PCA"),
    ("03_matriz_confusion.png", "Matriz de confusión"),
    ("04_curvas_roc.png", "Curvas ROC"),
    ("05_precision_recall.png", "Precision-Recall"),
    ("06_metricas_por_clase.png", "Métricas por clase"),
    ("07_validacion_cruzada.png", "Validación cruzada"),
    ("08_radar_metricas.png", "Radar de métricas"),
]

NOMBRES_DENSENET_PLOTS = [
    ("Figure_1.png", "Matriz de confusión — Cabeza B (7 clases)"),
    ("Figure_2.png", "Matriz de confusión — Cabeza A (binaria)"),
    ("Figure_3.png", "Curvas ROC — Cabeza B (One-vs-Rest)"),
    ("Figure_4.png", "Curvas Precision-Recall — Cabeza B"),
    ("Figure_5.png", "ROC + Precision-Recall — Cabeza A (binaria)"),
    ("Figure_6.png", "Métricas por clase — Cabeza B"),
    ("Figure_7.png", "Curvas de entrenamiento (loss / accuracy)"),
]


def galeria(modelo: dict, nombres: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(str(modelo["plots_dir"] / f), c) for f, c in nombres if (modelo["plots_dir"] / f).exists()]


# =============================================================================
# EDA — datos y figuras (estático, se calcula una sola vez al arrancar)
# =============================================================================
resumen = cargar_resumen_dict()
tests_eda = cargar_tests_estadisticos()

df_clases = cargar_hoja("distribucion_clases")
df_edad = cargar_hoja("estadist_edad").rename(columns={"Unnamed: 0": "grupo"})
df_mediana = cargar_hoja("mediana_edad_imputac")
df_sexo = cargar_hoja("distribucion_sexo")
df_loc = cargar_hoja("distribucion_localizac")
df_crosstab_loc = cargar_hoja("crosstab_loc_dx")
df_dxtype = cargar_hoja("distribucion_dxtype")
df_crosstab_dxtype = cargar_hoja("crosstab_dx_dxtype")
df_cramer = cargar_hoja("cramers_v")
df_grupos = cargar_hoja("gruposedad_dx")
df_full = cargar_hoja("dataset_limpio").reset_index(drop=True)
COLUMNAS_DATASET = df_full.columns.tolist()

kpi_md = (
    f"**Total imágenes:** {int(resumen['Total imágenes']):,} &nbsp;·&nbsp; "
    f"**Lesiones únicas:** {int(resumen['Total lesiones únicas']):,} &nbsp;·&nbsp; "
    f"**Lesiones con >1 imagen:** {int(resumen['Lesiones con >1 imagen']):,} &nbsp;·&nbsp; "
    f"**Clases dx:** {int(resumen['Clases diagnósticas (dx)'])} &nbsp;·&nbsp; "
    f"**Nulos sex:** {int(resumen['Nulos sex (sin imputar)'])} &nbsp;·&nbsp; "
    f"**Nulos localization:** {int(resumen['Nulos localization (sin imputar)'])}"
).replace(",", ".")

_kw = tests_eda.get("kruskal_wallis")
_pb = tests_eda.get("point_biserial")
_lineas_tests = []
if _kw:
    _lineas_tests.append(f"**Kruskal-Wallis (edad ~ dx):** H = {_kw['H']:.2f}, p = {_kw['p']:.3e}")
if _pb:
    _lineas_tests.append(f"**Point-Biserial (edad ~ malignidad):** r = {_pb['r']:.4f}, p = {_pb['p']:.3e}")
tests_md = "  \n".join(_lineas_tests) if _lineas_tests else "_No se encontraron los tests en el log._"

dx_opciones = sorted(df_full["dx"].dropna().unique().tolist())
sex_opciones = sorted(df_full["sex"].dropna().unique().tolist())
loc_opciones = sorted(df_full["localization"].dropna().unique().tolist())
edad_min_full, edad_max_full = int(df_full["age"].min()), int(df_full["age"].max())

galeria_eda = [(str(ruta_figura(f)), c) for f, c in FIGURAS_EDA if ruta_figura(f).exists()]

# El dataset limpio tiene 10.015 filas. Enviar la tabla completa al navegador
# en un solo gr.Dataframe pesa ~1 MB de JSON y bloquea el hilo principal al
# renderizar todas las celdas (esto es lo que provocaba el "la página no
# responde" al navegar). Por eso se limita a un máximo de filas visibles;
# los filtros siguen operando sobre el dataset completo, solo se trunca lo
# que se envía a la tabla.
MAX_FILAS_TABLA = 300


def _preparar_tabla(df):
    total = len(df)
    if total > MAX_FILAS_TABLA:
        texto = (
            f"**{total:,}** filas coinciden &nbsp;·&nbsp; mostrando las primeras "
            f"**{MAX_FILAS_TABLA}** (usa los filtros para acotar más)."
        ).replace(",", ".")
        return df.head(MAX_FILAS_TABLA), texto
    texto = f"**{total:,}** filas.".replace(",", ".")
    return df, texto


def filtrar_dataset(dx_sel, sex_sel, loc_sel, edad_desde, edad_hasta):
    # Nota: Gradio (en la versión instalada) no tiene un slider de rango de
    # dos manecillas (no existe gr.RangeSlider), así que el filtro de edad se
    # construye con dos gr.Slider independientes (desde / hasta).
    if edad_desde > edad_hasta:
        edad_desde, edad_hasta = edad_hasta, edad_desde
    df = df_full.copy()
    if dx_sel:
        df = df[df["dx"].isin(dx_sel)]
    if sex_sel:
        df = df[df["sex"].isin(sex_sel)]
    if loc_sel:
        df = df[df["localization"].isin(loc_sel)]
    df = df[df["age"].between(edad_desde, edad_hasta)]
    df = df.reset_index(drop=True)
    return _preparar_tabla(df)


def mostrar_imagen_seleccionada(evt: gr.SelectData):
    if evt is None or evt.row_value is None:
        return None, "Selecciona una celda de la tabla para ver la imagen de esa fila."
    fila = dict(zip(COLUMNAS_DATASET, evt.row_value))
    image_id = fila.get("image_id")
    ruta = buscar_imagen(image_id) if image_id else None
    detalle = "\n".join(f"- **{k}:** {v}" for k, v in fila.items())
    if ruta is None:
        return None, f"No se encontró `{image_id}.jpg` en las carpetas de imágenes configuradas.\n\n{detalle}"
    return str(ruta), f"### {image_id}\n\n{detalle}"


# =============================================================================
# Comparaciones — datos base (SVM y DenseNet con cabeza "7_clases" por defecto)
# =============================================================================
svm = cargar_svm("SVM")
svm_bal = cargar_svm("SVM_Balanceado")


def datos_densenet_comparacion(cabeza: str):
    dn = cargar_densenet("DenseNet", cabeza=cabeza)
    dn_bal = cargar_densenet("DenseNet_Balanceado", cabeza=cabeza)
    fig_barras = grafico_barras_comparativo(dn, dn_bal)
    fig_clase, aviso = graficos_por_clase_comparados(dn, dn_bal, metrica="f1_score")
    caption = (
        "DenseNet201 es multitarea: exporta una cabeza de diagnóstico específico (7 clases) "
        "y una cabeza de triaje binario en el mismo entrenamiento. Ambas versiones (original "
        f"y balanceada) comparten exactamente las mismas clases para la cabeza seleccionada "
        f"({', '.join(dn['clases'])})."
    )
    return (
        tabla_kpis_delta(dn, dn_bal),
        tabla_comparativa_global(dn, dn_bal),
        fig_barras,
        tabla_por_clase(dn),
        tabla_por_clase(dn_bal),
        fig_clase,
        aviso or "",
        caption,
    )


def datos_cross_comparacion(cabeza: str):
    dn_bal = cargar_densenet("DenseNet_Balanceado", cabeza=cabeza)
    fig_barras = grafico_barras_comparativo(svm_bal, dn_bal)
    if cabeza == "binaria":
        fig_clase, _ = graficos_por_clase_comparados(svm_bal, dn_bal, metrica="f1_score")
        nota = (
            "✅ Ambos modelos resuelven el mismo problema binario (Benigno vs. Maligno) "
            "sobre el conjunto balanceado, por lo que la comparación es directa."
        )
    else:
        fig_clase = None
        nota = (
            "⚠️ SVM_Balanceado es un clasificador **binario** (Benigno/Maligno), mientras que "
            "aquí se usa la cabeza de **7 clases** de DenseNet. Las métricas globales siguen "
            "siendo comparables, pero la tabla por clase no lo es (2 clases vs 7)."
        )
    return (
        tabla_kpis_delta(svm_bal, dn_bal),
        tabla_comparativa_global(svm_bal, dn_bal),
        fig_barras,
        tabla_por_clase(svm_bal),
        tabla_por_clase(dn_bal),
        fig_clase,
        nota,
    )


# =============================================================================
# Construcción de la interfaz
# =============================================================================
with gr.Blocks(title="HAM10000 — Comparación de modelos") as demo:
    gr.Markdown(
        "# 🔬 Comparación de modelos — HAM10000\n"
        "Panel interactivo (versión Gradio) para explorar el dataset y comparar el "
        "rendimiento de los modelos entrenados sobre HAM10000. Todos los datos se leen "
        "directamente de `outputs/` — no se recalcula ni se inventa ninguna métrica."
    )

    with gr.Tabs():
        # ---------------------------------------------------------------
        # 0. EDA
        # ---------------------------------------------------------------
        with gr.Tab("0. EDA"):
            gr.Markdown(kpi_md)

            with gr.Tabs():
                with gr.Tab("Distribución de clases"):
                    with gr.Row():
                        gr.Plot(value=fig_distribucion_clases(df_clases), scale=2)
                        gr.Plot(value=fig_malignidad_pie(df_clases), scale=1)
                    gr.Dataframe(value=df_clases, label="Distribución de clases (dx)", interactive=False)

                with gr.Tab("Edad"):
                    fila_global = df_edad[df_edad["grupo"] == "Global"].iloc[0]
                    gr.Markdown(
                        f"**Edad media:** {fila_global['mean']:.1f} años &nbsp;·&nbsp; "
                        f"**Desv. estándar:** {fila_global['std']:.1f} &nbsp;·&nbsp; "
                        f"**Mediana:** {fila_global['50%']:.0f} años &nbsp;·&nbsp; "
                        f"**Rango:** {fila_global['min']:.0f}–{fila_global['max']:.0f} años"
                    )
                    with gr.Row():
                        gr.Dataframe(value=df_edad, label="Estadística de edad por clase (dx)", interactive=False)
                        gr.Plot(value=fig_mediana_edad(df_mediana))
                    gr.Markdown(
                        f"Se imputaron {int(resumen['Nulos age (antes imputación)'])} valores nulos de edad "
                        "usando la mediana de edad de cada clase dx."
                    )

                with gr.Tab("Sexo y localización"):
                    with gr.Row():
                        gr.Plot(value=fig_sexo_pie(df_sexo), scale=1)
                        gr.Plot(value=fig_localizacion_barh(df_loc), scale=2)
                    gr.Plot(value=fig_heatmap_loc_dx(df_crosstab_loc))
                    gr.Markdown(
                        "ℹ️ La composición conjunta de sexo y localización (gráfico apilado) se "
                        "muestra en la pestaña 'Figuras generadas', ya que esa tabla cruzada no "
                        "forma parte de las hojas exportadas en el Excel."
                    )

                with gr.Tab("Tipo de diagnóstico"):
                    with gr.Row():
                        gr.Plot(value=fig_dxtype_bar(df_dxtype))
                        gr.Plot(value=fig_heatmap_dx_dxtype(df_crosstab_dxtype))

                with gr.Tab("Correlaciones"):
                    gr.Plot(value=fig_cramers_v(df_cramer))
                    gr.Markdown(tests_md)
                    gr.Markdown(
                        "_Cramér's V ≈ 1.00 entre `dx` y `malignidad` es esperable: la malignidad "
                        "se define directamente a partir de la clase diagnóstica, no es una "
                        "correlación encontrada de forma independiente._"
                    )

                with gr.Tab("Grupos de edad"):
                    gr.Plot(value=fig_grupos_edad(df_grupos))
                    gr.Dataframe(value=df_grupos, label="Grupo de edad × clase (dx)", interactive=False)

                with gr.Tab("Explorador de datos"):
                    gr.Markdown(
                        f"Dataset limpio completo: {len(df_full):,} filas × {df_full.shape[1]} columnas. "
                        "Filtra y haz clic en cualquier celda de una fila para ver la imagen de esa lesión."
                        .replace(",", ".")
                    )
                    if not carpetas_imagenes_disponibles():
                        gr.Markdown(
                            "⚠️ No se encontraron las carpetas de imágenes originales "
                            "(`HAM10000_images_part_1` / `_part_2`). La tabla funciona igual, "
                            "pero no se podrá previsualizar la imagen."
                        )

                    with gr.Row():
                        f_dx = gr.Dropdown(dx_opciones, multiselect=True, label="Clase (dx)")
                        f_sex = gr.Dropdown(sex_opciones, multiselect=True, label="Sexo")
                        f_loc = gr.Dropdown(loc_opciones, multiselect=True, label="Localización")
                    with gr.Row():
                        f_edad_desde = gr.Slider(edad_min_full, edad_max_full, value=edad_min_full, step=1, label="Edad desde")
                        f_edad_hasta = gr.Slider(edad_min_full, edad_max_full, value=edad_max_full, step=1, label="Edad hasta")
                        btn_filtrar = gr.Button("Aplicar filtros", variant="primary")

                    _tabla_inicial, _texto_inicial = _preparar_tabla(df_full)
                    texto_filas = gr.Markdown(_texto_inicial)
                    tabla_datos = gr.Dataframe(value=_tabla_inicial, label="Dataset limpio", interactive=False, max_height=420)

                    with gr.Row():
                        imagen_seleccionada = gr.Image(label="Imagen de la lesión", type="filepath")
                        detalle_seleccionado = gr.Markdown("Selecciona una celda de la tabla para ver la imagen de esa fila.")

                    btn_filtrar.click(
                        filtrar_dataset,
                        inputs=[f_dx, f_sex, f_loc, f_edad_desde, f_edad_hasta],
                        outputs=[tabla_datos, texto_filas],
                    )
                    tabla_datos.select(
                        mostrar_imagen_seleccionada,
                        inputs=None,
                        outputs=[imagen_seleccionada, detalle_seleccionado],
                    )

                with gr.Tab("Figuras generadas"):
                    gr.Gallery(value=galeria_eda, columns=2, height="auto", object_fit="contain")

        # ---------------------------------------------------------------
        # 1. SVM vs SVM Balanceado
        # ---------------------------------------------------------------
        with gr.Tab("1. SVM vs SVM Balanceado"):
            gr.Markdown(
                f"**Nota de comparabilidad:** SVM (baseline) resuelve un problema multiclase de "
                f"{svm['n_clases']} clases ({', '.join(svm['clases'])}), mientras que SVM_Balanceado "
                f"resuelve un problema **binario** ({', '.join(svm_bal['clases'])}) sobre un conjunto "
                f"rebalanceado. Las métricas globales son comparables porque usan la misma fórmula; "
                f"las tablas por clase se muestran por separado porque las clases no coinciden."
            )
            gr.Dataframe(value=tabla_kpis_delta(svm, svm_bal), label="Resumen (métricas clave)", interactive=False)
            with gr.Row():
                gr.Dataframe(value=tabla_comparativa_global(svm, svm_bal), label="Métricas globales", interactive=False)
                gr.Plot(value=grafico_barras_comparativo(svm, svm_bal))
            with gr.Row():
                gr.Dataframe(value=tabla_por_clase(svm), label=svm["nombre"], interactive=False)
                gr.Dataframe(value=tabla_por_clase(svm_bal), label=svm_bal["nombre"], interactive=False)

            cv_svm = svm.get("cv") or {}
            cv_svm_bal = svm_bal.get("cv") or {}
            gr.Markdown(
                f"**Validación cruzada — {svm['nombre']}:** Accuracy = "
                f"{cv_svm.get('cv_acc_mean', float('nan')):.4f} ± {cv_svm.get('cv_acc_std', float('nan')):.4f} "
                f"({cv_svm.get('n_folds', '?')} folds) &nbsp;·&nbsp; "
                f"**{svm_bal['nombre']}:** Accuracy = "
                f"{cv_svm_bal.get('cv_acc_mean', float('nan')):.4f} ± {cv_svm_bal.get('cv_acc_std', float('nan')):.4f} "
                f"({cv_svm_bal.get('n_folds', '?')} folds)"
            )

            with gr.Tabs():
                with gr.Tab(svm["nombre"]):
                    gr.Gallery(value=galeria(svm, NOMBRES_SVM_PLOTS), columns=2, height="auto", object_fit="contain")
                with gr.Tab(svm_bal["nombre"]):
                    gr.Gallery(value=galeria(svm_bal, NOMBRES_SVM_PLOTS), columns=2, height="auto", object_fit="contain")

        # ---------------------------------------------------------------
        # 2. DenseNet vs DenseNet Balanceado
        # ---------------------------------------------------------------
        with gr.Tab("2. DenseNet vs DenseNet Balanceado"):
            radio_cabeza_dn = gr.Radio(
                choices=[("Diagnóstico específico (7 clases)", "7_clases"), ("Triaje binario (Benigno / Maligno)", "binaria")],
                value="7_clases",
                label="Cabeza del modelo a comparar",
            )

            # Se calculan los valores iniciales ANTES de crear los componentes y se
            # pasan directamente como value=. No se usa demo.load() para rellenarlos
            # después: un gr.Plot que empieza vacío dentro de una pestaña oculta se
            # renderiza en un contenedor de tamaño 0 y se queda con el icono de
            # "cargando" para siempre aunque el dato llegue correctamente (bug de
            # Plotly con contenedores ocultos, no del callback en sí).
            (_kpi_dn0, _tg_dn0, _pb_dn0, _ta_dn0, _tb_dn0, _pc_dn0, _av_dn0, _cap_dn0) = datos_densenet_comparacion("7_clases")

            caption_dn = gr.Markdown(value=_cap_dn0)
            kpi_dn = gr.Dataframe(value=_kpi_dn0, label="Resumen (métricas clave)", interactive=False)
            with gr.Row():
                tabla_global_dn = gr.Dataframe(value=_tg_dn0, label="Métricas globales", interactive=False)
                plot_barras_dn = gr.Plot(value=_pb_dn0)
            with gr.Row():
                tabla_a_dn = gr.Dataframe(value=_ta_dn0, label="DenseNet", interactive=False)
                tabla_b_dn = gr.Dataframe(value=_tb_dn0, label="DenseNet Balanceado", interactive=False)
            aviso_dn = gr.Markdown(value=_av_dn0)
            plot_clase_dn = gr.Plot(value=_pc_dn0, label="Comparación por clase")

            dn_default = cargar_densenet("DenseNet")
            dn_bal_default = cargar_densenet("DenseNet_Balanceado")
            with gr.Tabs():
                with gr.Tab("DenseNet"):
                    gr.Gallery(value=galeria(dn_default, NOMBRES_DENSENET_PLOTS), columns=2, height="auto", object_fit="contain")
                with gr.Tab("DenseNet Balanceado"):
                    gr.Gallery(value=galeria(dn_bal_default, NOMBRES_DENSENET_PLOTS), columns=2, height="auto", object_fit="contain")

            salidas_dn = [kpi_dn, tabla_global_dn, plot_barras_dn, tabla_a_dn, tabla_b_dn, plot_clase_dn, aviso_dn, caption_dn]
            radio_cabeza_dn.change(datos_densenet_comparacion, inputs=[radio_cabeza_dn], outputs=salidas_dn)

        # ---------------------------------------------------------------
        # 3. SVM Balanceado vs DenseNet Balanceado
        # ---------------------------------------------------------------
        with gr.Tab("3. SVM Balanceado vs DenseNet Balanceado"):
            radio_cabeza_cross = gr.Radio(
                choices=[("Triaje binario (Benigno / Maligno) — comparable 1:1", "binaria"), ("Diagnóstico específico (7 clases) — no comparable por clase", "7_clases")],
                value="binaria",
                label="Cabeza de DenseNet a comparar contra el SVM Balanceado (binario)",
            )

            # Igual que en la pestaña 2: valores iniciales calculados antes de
            # crear los componentes (ver comentario allí sobre por qué no se usa
            # demo.load() para pestañas que no son la que se ve al abrir la página).
            (_kpi_cr0, _tg_cr0, _pb_cr0, _ta_cr0, _tb_cr0, _pc_cr0, _nota_cr0) = datos_cross_comparacion("binaria")

            nota_cross = gr.Markdown(value=_nota_cr0)
            kpi_cross = gr.Dataframe(value=_kpi_cr0, label="Resumen (métricas clave)", interactive=False)
            with gr.Row():
                tabla_global_cross = gr.Dataframe(value=_tg_cr0, label="Métricas globales", interactive=False)
                plot_barras_cross = gr.Plot(value=_pb_cr0)
            with gr.Row():
                tabla_a_cross = gr.Dataframe(value=_ta_cr0, label="SVM Balanceado", interactive=False)
                tabla_b_cross = gr.Dataframe(value=_tb_cr0, label="DenseNet Balanceado", interactive=False)
            plot_clase_cross = gr.Plot(value=_pc_cr0, label="Comparación por clase")

            with gr.Tabs():
                with gr.Tab("SVM Balanceado"):
                    gr.Gallery(value=galeria(svm_bal, NOMBRES_SVM_PLOTS), columns=2, height="auto", object_fit="contain")
                with gr.Tab("DenseNet Balanceado"):
                    gr.Gallery(value=galeria(dn_bal_default, NOMBRES_DENSENET_PLOTS), columns=2, height="auto", object_fit="contain")

            salidas_cross = [kpi_cross, tabla_global_cross, plot_barras_cross, tabla_a_cross, tabla_b_cross, plot_clase_cross, nota_cross]
            radio_cabeza_cross.change(datos_cross_comparacion, inputs=[radio_cabeza_cross], outputs=salidas_cross)


if __name__ == "__main__":
    # Las imágenes originales de HAM10000 viven fuera de esta carpeta
    # (D:\TFM\HAM10000\..., hermana de UNIR_TFM_HAM10000), y por seguridad
    # Gradio bloquea servir cualquier archivo devuelto por una función que
    # esté fuera del directorio de trabajo o del temporal del sistema. Hay
    # que autorizar explícitamente esas dos carpetas para que el visor de
    # imágenes del Explorador de datos funcione.
    demo.launch(allowed_paths=[str(d) for d in IMAGENES_DIRS])
