import streamlit as st
from modelo import load_dataset, recomendar_peliculas_multiples, train_model
from grafico import generar_df_comparacion, obtener_perfil_contenido
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="Recomendador de Películas")

st.title("Recomendador de Películas")
st.markdown("""
En esta aplicacion podes buscar una o mas películas por su título, y si hay varias coincidencias podes 
identificar la tuya segun su director.
Con todas las peliculas que agregues, vamos a recomendarte todas las peliculas que vos quieras que 
pensamos que te gustarian ver.
""")

# Carga del dataset y modelo
with st.spinner("Cargando dataset..."):
    df = load_dataset()
with st.spinner("Entrenando modelo..."):
    train_model()

# Estado para guardar índices seleccionados
if "peliculas_idx" not in st.session_state:
    st.session_state.peliculas_idx = []

if "recomendadas" not in st.session_state:
    st.session_state.recomendadas = None

st.subheader("Buscar película")
titulo = st.text_input("Escribí el título de una película:")

# Se va buscando pelicula por pelicula, y cada vez que agregue una al peliculas_idx
# va apareciendo en el listado de abajo
if titulo:
    # resulta en un dataframe
    coincidencias = df[df["title"].str.contains(titulo, case=False, na=False)]

    if coincidencias.empty:
        st.warning("❌ No se encontró ninguna película con ese título.")
    else:
        if len(coincidencias) > 1:
            opciones = [
                f"{row['title']} ({row['release_date']}) - {row['directors']}"
                for _, row in coincidencias.iterrows()
            ]
            seleccion = st.selectbox(
                "Se encontraron varias coincidencias, elegí una:",
                opciones,
                key=f"select_{titulo}"
            )
            idx_pelicula = coincidencias.index[opciones.index(seleccion)]
        else:
            idx_pelicula = coincidencias.index[0]
            st.info(f"Se seleccionó: {df.loc[idx_pelicula, 'title']}")
            
        poster_url = df.loc[idx_pelicula, "poster_path"]
        if isinstance(poster_url, str) and poster_url.strip() != "":
            st.image(f"https://image.tmdb.org/t/p/w300{poster_url}", width=180)
        else:
            st.write("(Sin póster disponible)")

        if st.button("+ Agregar película", key=f"add_{idx_pelicula}"):
            if idx_pelicula not in st.session_state.peliculas_idx:
                st.session_state.peliculas_idx.append(idx_pelicula)
                st.success(f"'{df.loc[idx_pelicula, 'title']}' agregada a la lista de recomendacion.")
            else:
                st.info("Esa película ya está en la lista.")

# peliculas que se van a usar para la recomendacion
if st.session_state.peliculas_idx:
    st.subheader("Películas base para recomendación:")

    # Mostrar cada película con título + poster
    cols = st.columns(len(st.session_state.peliculas_idx))

    for col, idx in zip(cols, st.session_state.peliculas_idx):
        with col:
            titulo = df.loc[idx, "title"]
            director = df.loc[idx, "directors"]
            poster_url = df.loc[idx, "poster_path"]

            st.markdown(f"**{titulo}**")
            if isinstance(poster_url, str) and poster_url.strip() != "":
                st.image(f"https://image.tmdb.org/t/p/w300{poster_url}", width=180)
            else:
                st.write("(Sin póster disponible)")

            st.caption(f"Director/es: {director}")

    if st.button("X - Limpiar lista"):
        st.session_state.peliculas_idx = []
        st.info("Lista vaciada.")

    n_recomendadas = st.number_input(
        "Cantidad de películas a recomendar:",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

    if st.button("Generar recomendaciones"):
        with st.spinner("Buscando películas similares..."):
            st.session_state.recomendadas = recomendar_peliculas_multiples(
                st.session_state.peliculas_idx,
                n=n_recomendadas
            )


    if st.session_state.recomendadas is None or st.session_state.recomendadas.empty:
        st.error("No se pudieron generar recomendaciones.")
    else:
        base_titles = [df.loc[idx, "title"] for idx in st.session_state.peliculas_idx]
        st.success(f"Películas similares a **{', '.join(base_titles)}**:")
        columnas_mostradas = ["title", "release_date", "original_language", "budget", "revenue", "runtime", "vote_average", "vote_count", "genres", "similitud_promedio"]
        st.dataframe(st.session_state.recomendadas[columnas_mostradas])

        # --- GRÁFICO INTERACTIVO ALTAR ---
        st.subheader("Comparación de Features TF-IDF")

        # Construir perfiles para la película base seleccionada y las películas
        # que aparecen en la tabla de recomendaciones actual (`st.session_state.recomendadas`).
        # Esto asegura que el selector de comparación muestre exactamente las mismas
        # películas que el recomendador enseñó.
        n_features = 15

        # Selector de base: elegir entre las películas que el usuario agregó como base
        bases = st.session_state.peliculas_idx
        seleccion_base = st.selectbox(
            "Seleccioná la película base para comparar:",
            options=bases,
            format_func=lambda x: df.loc[x, 'title']
        )

        # Perfil TF-IDF de la base
        perfil_base = obtener_perfil_contenido(seleccion_base, n_features=n_features)
        perfil_base['comparison_group'] = f"Base: {df.loc[seleccion_base, 'title']}"
        perfil_base['base_index'] = seleccion_base

        # Perfiles de las películas recomendadas (usar 'orig_index' para mapear al DF original)
        perfiles_recomendadas = []
        etiquetas_recomendadas = []
        if st.session_state.recomendadas is not None and not st.session_state.recomendadas.empty:
            for _, row in st.session_state.recomendadas.iterrows():
                # 'orig_index' fue añadido en modelo.recomendar_peliculas_multiples
                orig_idx = row.get('orig_index') if 'orig_index' in row.index else None
                if orig_idx is None:
                    # Si no existe el mapeo original, intentar buscar por título como fallback
                    try:
                        orig_idx = df[(df['title'] == row['title']) & (df['release_date'] == row.get('release_date'))].index[0]
                    except Exception:
                        continue

                perfil_rec = obtener_perfil_contenido(int(orig_idx), n_features=n_features)
                etiqueta = f"Recom: {row['title']}"
                perfil_rec['comparison_group'] = etiqueta
                perfil_rec['base_index'] = seleccion_base
                perfiles_recomendadas.append(perfil_rec)
                etiquetas_recomendadas.append(etiqueta)

        if len(perfiles_recomendadas) == 0:
            st.info("No hay recomendaciones para comparar con la base seleccionada.")
        else:
            # Concatenar base + recomendaciones para el gráfico
            df_para_grafico = pd.concat([perfil_base] + perfiles_recomendadas, ignore_index=True)

            # Selector de recomendada (usar las mismas que aparecen en la tabla)
            seleccion_comparacion = st.selectbox(
                "Seleccionar película recomendada para comparar y ver póster:",
                options=etiquetas_recomendadas,
                index=0,
                key=f"comp_select_{seleccion_base}"
            )

            # Mostrar póster de la película seleccionada para comparar
            pelicula_comparada = None
            for _, row in st.session_state.recomendadas.iterrows():
                if f"Recom: {row['title']}" == seleccion_comparacion:
                    pelicula_comparada = row
                    break
            
            if pelicula_comparada is not None:
                poster_url = pelicula_comparada.get("poster_path", "")
                titulo = pelicula_comparada["title"]
                st.markdown(f"### {titulo}")
                if isinstance(poster_url, str) and poster_url.strip() != "":
                    st.image(f"https://image.tmdb.org/t/p/w300{poster_url}", width=250)
                else:
                    st.write("(Sin póster disponible)")

            df_plot = df_para_grafico[df_para_grafico['comparison_group'].isin([perfil_base['comparison_group'].iloc[0], seleccion_comparacion])]

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x='peso_tfidf:Q',
                    y=alt.Y('feature:N', sort=df_plot.groupby('feature')['peso_tfidf'].sum().sort_values(ascending=False).index.tolist()),
                    color='comparison_group:N',
                    tooltip=['feature', 'peso_tfidf', 'comparison_group']
                )
                .properties(title='Comparación de Features TF-IDF entre Base y Recomendada')
            )

            st.altair_chart(chart, use_container_width=True)
