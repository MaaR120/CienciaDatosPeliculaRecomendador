import streamlit as st
from modelo import load_dataset, recomendar_peliculas_multiples, train_model
from grafico import generar_df_comparacion, obtener_perfil_contenido
import pandas as pd
import altair as alt
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Recomendador de Películas")

st.title("Recomendador de Películas")
st.markdown("""
En esta aplicacion podes buscar una o mas películas por su título, y si hay varias coincidencias podes 
identificar la tuya segun su director.
Con todas las peliculas que agregues, vamos a recomendarte todas las peliculas que vos quieras que 
pensamos que te gustarian ver.
""")


def render_poster_con_fullscreen(poster_path, width=180, alt="Poster"):
        """Renderiza un poster con un botón en la esquina superior derecha que abre la imagen en tamaño original.

        poster_path: ruta en TMDB (ej: '/abc.jpg')
        width: ancho en píxeles del thumbnail
        """
        if not isinstance(poster_path, str) or poster_path.strip() == "":
                st.write("(Sin póster disponible)")
                return

        thumb = f"https://image.tmdb.org/t/p/w300{poster_path}"
        full = f"https://image.tmdb.org/t/p/original{poster_path}"

        # HTML inline: imagen dentro de un contenedor relativo con enlace absoluto en la esquina
        html = f'''<div style="position:relative; display:inline-block;">
            <img src="{thumb}" alt="{alt}" style="width:{width}px; border-radius:6px; display:block;"/>
            <a href="{full}" target="_blank" rel="noopener noreferrer" style="position:absolute; top:6px; right:6px; background:rgba(0,0,0,0.55); color:white; padding:6px; border-radius:6px; text-decoration:none; font-weight:600;">🔍</a>
        </div>'''

        # Usar components.html para evitar problemas de sanitización y controlar altura mínima
        # Calculamos una altura aproximada en px para el iframe (anchura * 1.5) como margen
        height = int(width * 1.6)
        components.html(html, height=height)


# Carga del dataset y modelo
with st.spinner("Cargando dataset..."):
    df = load_dataset()
with st.spinner("Entrenando modelo..."):
    train_model()
    
st.header("Explorador del Dataset Completo")

with st.expander("Mostrar / buscar películas en el dataset"):
    st.markdown("""
    Acá podés buscar películas dentro de todo el dataset según distintos filtros.
    Por motivos de rendimiento, se muestra un máximo de 1000 resultados por vez.
    """)

    # --- Filtros ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filtro_titulo = st.text_input("Título contiene:")
    with col2:
        filtro_director = st.text_input("Director contiene:")
    with col3:
        filtro_genero = st.text_input("Género contiene:")
    with col4:
        filtro_idioma = st.text_input("Idioma original contiene:")

    # --- Aplicar filtros dinámicamente ---
    df_filtrado = df.copy()

    if filtro_titulo:
        df_filtrado = df_filtrado[df_filtrado["title"].str.contains(filtro_titulo, case=False, na=False)]
    if filtro_director:
        df_filtrado = df_filtrado[df_filtrado["directors"].str.contains(filtro_director, case=False, na=False)]
    if filtro_genero:
        df_filtrado = df_filtrado[df_filtrado["genres"].str.contains(filtro_genero, case=False, na=False)]
    if filtro_idioma:
        df_filtrado = df_filtrado[df_filtrado["original_language"].astype(str).str.contains(filtro_idioma, case=False, na=False)]

    # --- Selección de cantidad de filas y paginación ---
    total_resultados = len(df_filtrado)
    st.write(f"{total_resultados:,} películas encontradas. Mostrando hasta 1000 resultados:")

    max_mostrar = 1000
    df_mostrar = df_filtrado.head(max_mostrar)

    # --- Mostrar dataframe paginado ---
    st.dataframe(
        df_mostrar.drop(["poster_path", "status", "adult"], errors="ignore"),
        use_container_width=True,
        height=500
    )

    # --- Opción de descarga ---
    csv = df_mostrar.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar resultados como CSV",
        data=csv,
        file_name="peliculas_filtradas.csv",
        mime="text/csv"
    )

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
            # Mostrar poster con botón de pantalla completa en la esquina superior derecha
            render_poster_con_fullscreen(poster_url, width=180, alt=df.loc[idx_pelicula, 'title'])
        else:
            st.write("(Sin póster disponible)")

        if st.button("+ Agregar película", key=f"add_{idx_pelicula}"):
            if idx_pelicula not in st.session_state.peliculas_idx:
                st.session_state.peliculas_idx.append(idx_pelicula)
                st.success(f"'{df.loc[idx_pelicula, 'title']}' agregada a la lista de recomendacion.")
                st.session_state.recomendadas = None  # Limpiar recomendaciones anteriores
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
                render_poster_con_fullscreen(poster_url, width=180, alt=titulo)
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

    # Opciones avanzadas de recomendación
    with st.expander("Opciones avanzadas de recomendación"):
        st.markdown("""
        ### Método de Cálculo de Similitud
        
        Elige cómo se combinan las similitudes de múltiples películas base:
        """)
        usar_mediana = st.checkbox(
            "Usar mediana en lugar de promedio ponderado", 
            help="""
            - Con promedio ponderado (default): Las películas recomendadas serán más similares a todas las películas base, 
              usando los pesos definidos abajo para dar más o menos importancia a cada película.
            - Con mediana: Las recomendaciones se basan en el valor central de similitud para cada película recomendada, 
              ignorando los pesos. Útil cuando las películas base son muy diferentes entre sí.
            """
        )
        
        explicar = st.checkbox(
            "Explicar recomendaciones", 
            value=True,
            help="""
            Muestra detalles de por qué se recomienda cada película:
            - Géneros en común con cada película base
            - Directores compartidos
            - Actores compartidos
            """
        )
        
        # Pesos personalizados para cada película base
        st.markdown("""
        ### Importancia Relativa
        
        Ajusta el peso (importancia) de cada película base en las recomendaciones:
        - 1.0: Máxima influencia
        - 0.0: Sin influencia
        
        ⚠️ Nota: Los pesos solo se utilizan cuando se usa el promedio ponderado. 
        Si se selecciona "Usar mediana", los pesos se ignoran.
        """)
        pesos = []
        cols_pesos = st.columns(len(st.session_state.peliculas_idx))
        
        # Si se usa mediana, mostrar los sliders deshabilitados con valor 1.0
        if usar_mediana:
            for i, (col, idx) in enumerate(zip(cols_pesos, st.session_state.peliculas_idx)):
                with col:
                    st.slider(
                        f"Peso para {df.loc[idx, 'title']}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        step=0.1,
                        key=f"peso_{idx}_disabled",
                        disabled=True,
                        help="Los pesos no se utilizan cuando se usa la mediana"
                    )
            # Usar pesos iguales (1.0) para todas las películas
            pesos = [1.0] * len(st.session_state.peliculas_idx)
        else:
            for i, (col, idx) in enumerate(zip(cols_pesos, st.session_state.peliculas_idx)):
                with col:
                    peso = st.slider(
                        f"Peso para {df.loc[idx, 'title']}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        step=0.1,
                        key=f"peso_{idx}",
                        help="Desliza para ajustar cuánto influye esta película en las recomendaciones"
                    )
                    pesos.append(peso)

    # Validar que al menos un peso sea mayor que cero
    if sum(pesos) == 0:
        st.error("❌ Error: Al menos una película debe tener un peso mayor que cero.")
    elif st.button("Generar recomendaciones"):
        # Validar que al menos un peso sea mayor que cero antes de intentar recomendar
        if sum(pesos) == 0:
            st.error("❌ Al menos una película debe tener un peso mayor que cero.")
        else:
            with st.spinner("Buscando películas similares..."):
                try:
                    st.session_state.recomendadas = recomendar_peliculas_multiples(
                        st.session_state.peliculas_idx,
                        n=n_recomendadas,
                        pesos=pesos,
                        usar_mediana=usar_mediana,
                        explicar=explicar
                    )
                except Exception as e:
                    st.error(f"❌ Error al generar recomendaciones: {str(e)}")
                    st.session_state.recomendadas = None

    if st.session_state.recomendadas is None or st.session_state.recomendadas.empty:
        st.error("No se pudieron generar recomendaciones.")
    else:
        base_titles = [df.loc[idx, "title"] for idx in st.session_state.peliculas_idx]
        st.success(f"Películas similares a **{', '.join(base_titles)}**:")
        
        # Columnas base
        columnas_mostradas = ["title", "release_date", "original_language", "genres", "similitud_promedio"]
        
        # Añadir columnas de similitud individual si hay más de una película base
        if len(st.session_state.peliculas_idx) > 1:
            columnas_similitud = [col for col in st.session_state.recomendadas.columns if col.startswith('similitud_con_')]
            columnas_mostradas.extend(columnas_similitud)
        
        df_mostrado = st.session_state.recomendadas[columnas_mostradas].copy()
        
        # Formatear similitudes como porcentajes
        for col in df_mostrado.columns:
            if col.startswith('similitud'):
                df_mostrado[col] = df_mostrado[col].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df_mostrado)
        
        # Mostrar explicaciones si están disponibles
        if 'explicacion' in st.session_state.recomendadas.columns:
            st.subheader("Explicación de Recomendaciones")
            for idx, row in st.session_state.recomendadas.iterrows():
                with st.expander(f"Por qué recomendamos '{row['title']}'"):
                    explicacion = row['explicacion']
                    
                    # Sección de géneros
                    st.markdown("### 🎭 Géneros en Común")
                    generos_info = explicacion.get('generos', {})
                    if generos_info:
                        for pelicula, generos in generos_info.items():
                            if generos:
                                st.markdown(f"**Con {pelicula}:**")
                                for genero in sorted(generos):
                                    st.markdown(f"- {genero}")
                                st.markdown("")
                            else:
                                st.markdown(f"**Con {pelicula}:** Ninguno")
                    else:
                        st.markdown("No se encontraron géneros en común.")

                    # Keywords compartidas
                    st.markdown("### 🔑 Keywords en Común")
                    kw_info = explicacion.get('keywords', {})
                    if kw_info:
                        for pelicula, kws in kw_info.items():
                            if kws:
                                st.markdown(f"**Con {pelicula}:** {', '.join(kws)}")
                            else:
                                st.markdown(f"**Con {pelicula}:** Ninguna")
                    else:
                        st.markdown("No se encontraron keywords en común.")

                    # Production companies compartidas
                    st.markdown("### � Production Companies en Común")
                    pc_info = explicacion.get('production_companies', {})
                    if pc_info:
                        for pelicula, pcs in pc_info.items():
                            if pcs:
                                st.markdown(f"**Con {pelicula}:** {', '.join(pcs)}")
                            else:
                                st.markdown(f"**Con {pelicula}:** Ninguna")
                    else:
                        st.markdown("No se encontraron production companies en común.")

                    # Directores y actores
                    if explicacion.get('directores'):
                        st.markdown("### 🎬 Directores en Común")
                        for director in explicacion.get('directores', []):
                            st.markdown(f"- {director}")

                    if explicacion.get('actores'):
                        st.markdown("### 🎭 Actores en Común")
                        for actor in explicacion.get('actores', []):
                            st.markdown(f"- {actor}")

                    # Top TF-IDF terms que explican la similitud
                    top_terms = explicacion.get('top_terms', [])
                    if top_terms:
                        st.markdown("### ✨ Términos TF‑IDF que explican la similitud")
                        st.markdown(", ".join(top_terms))

                    # Contribución relativa de cada película base
                    contribs = explicacion.get('contribuciones', {})
                    if contribs:
                        st.markdown("### 📊 Contribución relativa de cada película base")
                        for base, val in contribs.items():
                            st.markdown(f"- {base}: {val*100:.1f}%")

                    # Mostrar similitudes individuales si están disponibles
                    similitudes_ind = explicacion.get('similitudes_individuales', {})
                    if similitudes_ind:
                        st.markdown("### 🔍 Similitudes individuales (por base)")
                        for base, sim in similitudes_ind.items():
                            st.markdown(f"- {base}: {sim*100:.1f}%")

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
                    render_poster_con_fullscreen(poster_url, width=250, alt=titulo)
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
