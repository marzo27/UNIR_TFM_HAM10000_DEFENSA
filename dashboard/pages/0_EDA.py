import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit no añade automáticamente la raíz de la app al sys.path al
# ejecutar páginas dentro de pages/, así que lo hacemos explícito aquí
# para poder importar el paquete utils/ compartido entre las páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.carga_eda import FIGURAS, cargar_hoja, cargar_resumen_dict, cargar_tests_estadisticos, ruta_figura
from utils.carga_imagenes import buscar_imagen, carpetas_imagenes_disponibles
from utils.render import galeria_imagenes

st.set_page_config(page_title="EDA — HAM10000", layout="wide", page_icon="🔎")
st.title("🔎 Análisis Exploratorio de Datos (EDA) — HAM10000")
st.caption(
    "Fuente: `outputs/EDA/eda_resultados_HAM10000.xlsx` (12 hojas) y "
    "`outputs/EDA/eda_ham10000_v1_salida.txt`. Todos los valores se leen "
    "directamente de esos archivos, sin recalcular ni inventar nada."
)

resumen = cargar_resumen_dict()

# ---------------------------------------------------------------------------
# KPIs generales (hoja 01_Resumen_General)
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total imágenes", f"{int(resumen['Total imágenes']):,}".replace(",", "."))
c2.metric("Lesiones únicas", f"{int(resumen['Total lesiones únicas']):,}".replace(",", "."))
c3.metric("Lesiones con >1 imagen", f"{int(resumen['Lesiones con >1 imagen']):,}".replace(",", "."))
c4.metric("Clases diagnósticas (dx)", int(resumen["Clases diagnósticas (dx)"]))
c5.metric("Nulos age (tras imputación)", int(resumen["Nulos age (después imputación)"]))

c1, c2, c3 = st.columns(3)
c1.metric("Nulos sex (sin imputar)", int(resumen["Nulos sex (sin imputar)"]))
c2.metric("Nulos localization (sin imputar)", int(resumen["Nulos localization (sin imputar)"]))
c3.metric("Columnas originales", int(resumen["Columnas originales"]))

st.divider()

tabs = st.tabs(
    [
        "Distribución de clases",
        "Edad",
        "Sexo y localización",
        "Tipo de diagnóstico",
        "Correlaciones",
        "Grupos de edad",
        "Explorador de datos",
        "Figuras generadas",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 — Distribución de clases (hoja 02)
# ---------------------------------------------------------------------------
with tabs[0]:
    df_clases = cargar_hoja("distribucion_clases")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.bar(
            df_clases.sort_values("n_imagenes", ascending=False),
            x="dx", y="n_imagenes", color="malignidad", text="porcentaje_%",
            labels={"dx": "Clase (dx)", "n_imagenes": "N.º de imágenes", "malignidad": "Malignidad"},
            title="Distribución de imágenes por clase diagnóstica",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")
    with c2:
        resumen_malignidad = df_clases.groupby("malignidad", as_index=False)[["n_imagenes", "porcentaje_%"]].sum()
        fig_pie = px.pie(
            resumen_malignidad, names="malignidad", values="n_imagenes",
            title="Benigno vs. Maligno (global)", hole=0.45,
        )
        st.plotly_chart(fig_pie, width="stretch")

    st.dataframe(df_clases, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2 — Edad (hojas 03 y 04)
# ---------------------------------------------------------------------------
with tabs[1]:
    df_edad = cargar_hoja("estadist_edad").rename(columns={"Unnamed: 0": "grupo"})
    df_mediana = cargar_hoja("mediana_edad_imputac")

    fila_global = df_edad[df_edad["grupo"] == "Global"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Edad media (global)", f"{fila_global['mean']:.1f} años")
    c2.metric("Desv. estándar", f"{fila_global['std']:.1f} años")
    c3.metric("Mediana", f"{fila_global['50%']:.0f} años")
    c4.metric("Rango", f"{fila_global['min']:.0f} – {fila_global['max']:.0f} años")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Estadística de edad por clase (dx)")
        st.dataframe(df_edad, width="stretch", hide_index=True)
    with c2:
        st.subheader("Mediana de edad usada para imputar valores nulos")
        fig = px.bar(
            df_mediana.sort_values("mediana_edad_imputada", ascending=False),
            x="dx", y="mediana_edad_imputada", text="mediana_edad_imputada",
            hover_data=["nombre_largo"],
            labels={"dx": "Clase (dx)", "mediana_edad_imputada": "Mediana de edad (años)"},
        )
        st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Se imputaron {int(resumen['Nulos age (antes imputación)'])} valores nulos de edad "
        "usando la mediana de edad de cada clase dx (tabla de la derecha)."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Sexo y localización (hojas 05, 06, 09)
# ---------------------------------------------------------------------------
with tabs[2]:
    df_sexo = cargar_hoja("distribucion_sexo").fillna({"sexo": "Sin dato"})
    df_loc = cargar_hoja("distribucion_localizac").fillna({"localizacion": "Sin dato"})
    df_crosstab_loc = cargar_hoja("crosstab_loc_dx")

    c1, c2 = st.columns([1, 1.6])
    with c1:
        fig = px.pie(df_sexo, names="sexo", values="conteo", title="Distribución por sexo", hole=0.45)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(
            df_loc.sort_values("conteo", ascending=True),
            x="conteo", y="localizacion", orientation="h", text="porcentaje_%",
            labels={"conteo": "N.º de imágenes", "localizacion": "Localización"},
            title="Distribución por localización anatómica",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Localización × clase diagnóstica (dx)")
    matriz = df_crosstab_loc.set_index("localization")
    fig = px.imshow(
        matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Clase (dx)", y="Localización", color="N.º imágenes"),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "La composición conjunta de sexo y localización (gráfico apilado) se muestra "
        "en la figura pre-generada de la pestaña 'Figuras generadas', ya que esa "
        "tabla cruzada no forma parte de las hojas exportadas en el Excel."
    )

# ---------------------------------------------------------------------------
# Tab 4 — Tipo de diagnóstico (hojas 07, 08)
# ---------------------------------------------------------------------------
with tabs[3]:
    df_dxtype = cargar_hoja("distribucion_dxtype")
    df_crosstab_dxtype = cargar_hoja("crosstab_dx_dxtype")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = px.bar(
            df_dxtype.sort_values("conteo", ascending=False),
            x="dx_type", y="conteo", text="porcentaje_%", hover_data=["nombre_largo"],
            labels={"dx_type": "Tipo de diagnóstico", "conteo": "N.º de imágenes"},
            title="Método de confirmación diagnóstica",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")
    with c2:
        matriz = df_crosstab_dxtype.set_index("dx")
        fig = px.imshow(
            matriz, text_auto=True, color_continuous_scale="Purples", aspect="auto",
            labels=dict(x="Tipo de diagnóstico", y="Clase (dx)", color="N.º imágenes"),
            title="Clase (dx) × tipo de diagnóstico",
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5 — Correlaciones (hoja 10 + tests del .txt)
# ---------------------------------------------------------------------------
with tabs[4]:
    df_cramer = cargar_hoja("cramers_v").rename(columns={"Unnamed: 0": "variable"}).set_index("variable")

    fig = px.imshow(
        df_cramer, text_auto=".3f", color_continuous_scale="RdPu", zmin=0, zmax=1,
        labels=dict(color="Cramér's V"),
        title="Matriz de correlación — Cramér's V (variables categóricas)",
    )
    st.plotly_chart(fig, width="stretch")

    tests = cargar_tests_estadisticos()
    c1, c2 = st.columns(2)
    with c1:
        kw = tests.get("kruskal_wallis")
        if kw:
            st.metric("Kruskal-Wallis (edad ~ dx) — H", f"{kw['H']:.2f}")
            st.caption(f"p-valor = {kw['p']:.3e}")
        else:
            st.warning("No se encontró el resultado de Kruskal-Wallis en el log.")
    with c2:
        pb = tests.get("point_biserial")
        if pb:
            st.metric("Point-Biserial (edad ~ malignidad) — r", f"{pb['r']:.4f}")
            st.caption(f"p-valor = {pb['p']:.3e}")
        else:
            st.warning("No se encontró el resultado de Point-Biserial en el log.")

    st.caption(
        "Cramér's V ≈ 1.00 entre `dx` y `malignidad` es esperable: la malignidad se "
        "define directamente a partir de la clase diagnóstica, no es una correlación "
        "encontrada de forma independiente."
    )

# ---------------------------------------------------------------------------
# Tab 6 — Grupos de edad (hoja 11)
# ---------------------------------------------------------------------------
with tabs[5]:
    df_grupos = cargar_hoja("gruposedad_dx").set_index("age_group")
    fig = px.imshow(
        df_grupos, text_auto=True, color_continuous_scale="Oranges", aspect="auto",
        labels=dict(x="Clase (dx)", y="Grupo de edad", color="N.º imágenes"),
        title="Grupo de edad × clase diagnóstica (dx)",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df_grupos.reset_index(), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 7 — Explorador de datos (hoja 12, dataset limpio completo)
# ---------------------------------------------------------------------------
with tabs[6]:
    df_full = cargar_hoja("dataset_limpio")

    st.caption(f"Dataset limpio completo: {len(df_full):,} filas × {df_full.shape[1]} columnas.".replace(",", "."))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        f_dx = st.multiselect("Clase (dx)", sorted(df_full["dx"].dropna().unique()))
    with c2:
        f_sex = st.multiselect("Sexo", sorted(df_full["sex"].dropna().unique()))
    with c3:
        f_loc = st.multiselect("Localización", sorted(df_full["localization"].dropna().unique()))
    with c4:
        edad_min, edad_max = int(df_full["age"].min()), int(df_full["age"].max())
        rango_edad = st.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))

    df_filtrado = df_full.copy()
    if f_dx:
        df_filtrado = df_filtrado[df_filtrado["dx"].isin(f_dx)]
    if f_sex:
        df_filtrado = df_filtrado[df_filtrado["sex"].isin(f_sex)]
    if f_loc:
        df_filtrado = df_filtrado[df_filtrado["localization"].isin(f_loc)]
    df_filtrado = df_filtrado[df_filtrado["age"].between(*rango_edad)]

    st.write(f"**{len(df_filtrado):,}** filas tras filtrar.".replace(",", "."))

    if not carpetas_imagenes_disponibles():
        st.warning(
            "No se encontraron las carpetas de imágenes originales "
            "(`D:\\TFM\\HAM10000\\HAM10000_images_part_1` / `_part_2`). "
            "La tabla se muestra igualmente, pero no se podrá previsualizar la imagen."
        )

    st.caption("Selecciona (marca la casilla de) una fila para ver la imagen de esa lesión.")
    df_mostrar = df_filtrado.reset_index(drop=True)
    evento = st.dataframe(
        df_mostrar,
        width="stretch",
        height=400,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    filas_seleccionadas = evento.selection.rows if evento and evento.selection else []
    if filas_seleccionadas:
        fila = df_mostrar.iloc[filas_seleccionadas[0]]
        image_id = fila["image_id"]
        ruta_img = buscar_imagen(image_id)

        st.subheader(f"Imagen: {image_id}")
        c1, c2 = st.columns([1, 2])
        with c1:
            if ruta_img is not None:
                st.image(str(ruta_img), caption=f"{image_id}.jpg", width="stretch")
            else:
                st.error(
                    f"No se encontró `{image_id}.jpg` en las carpetas de imágenes configuradas."
                )
        with c2:
            # Se castea a str: la fila mezcla texto y números en una misma
            # columna "Valor", lo que rompe la serialización a Arrow de
            # st.dataframe (ArrowTypeError) si se deja como dtype object.
            detalle = fila.to_frame(name="Valor").astype(str)
            st.dataframe(detalle, width="stretch")

    if not df_filtrado.empty:
        fig = px.histogram(
            df_filtrado, x="dx", color="malignidad",
            title="Distribución de clases en la selección filtrada",
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 8 — Figuras generadas por el script de EDA
# ---------------------------------------------------------------------------
with tabs[7]:
    galeria_imagenes("", [(ruta_figura(f), c) for f, c in FIGURAS])
