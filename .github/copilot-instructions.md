# Instrucciones para Agentes AI - Sistema Recomendador de Películas

## Arquitectura y Componentes Principales

Este proyecto es un sistema de recomendación de películas que utiliza técnicas de procesamiento de lenguaje natural (NLP) y análisis de similitud de contenido. Los componentes principales son:

- `modelo.py`: Core del sistema de recomendación
  - Implementa el pipeline de procesamiento de datos y el modelo de recomendación
  - Utiliza TF-IDF y Nearest Neighbors para encontrar películas similares
  
- `streamlit_app.py`: Interfaz de usuario web
  - Maneja la interacción con el usuario y visualización de resultados
  - Gestiona el estado de la aplicación usando `st.session_state`

- `grafico.py`: Visualización y análisis de features
  - Genera comparaciones interactivas de características TF-IDF
  - Produce gráficos Altair para exploración de similitudes

## Patrones y Convenciones

### Cacheo de Datos y Modelo
- Uso de decoradores `@st.cache_data` para dataset y `@st.cache_resource` para modelo
- El modelo se entrena una sola vez y se reutiliza en toda la sesión

### Pipeline de Procesamiento
1. Limpieza de texto con `TextContentCleaner`
2. Creación de "sopa de palabras" con `SoupCreator` 
3. Transformación TF-IDF y cálculo de similitudes

### Manejo de Estado
- Películas seleccionadas almacenadas en `st.session_state.peliculas_idx`
- Recomendaciones guardadas en `st.session_state.recomendadas`

## Flujos de Desarrollo

### Setup del Entorno
```bash
pip install -r requirements.txt
```

### Ejecución Local
```bash
streamlit run streamlit_app.py
```

### Dataset
- Se descarga automáticamente de Google Drive al iniciar
- Formato CSV con separador ";" 
- Incluye metadata de películas (título, género, actores, director, etc.)

## Puntos de Integración

### APIs Externas
- TMDB API para posters de películas (`https://image.tmdb.org/t/p/w300`)
- Google Drive para dataset (`gdown` para descarga)

### Dependencias Clave
- `streamlit`: UI y manejo de estado
- `scikit-learn`: Procesamiento y modelo
- `altair`: Visualizaciones interactivas

## Notas Importantes
- El sistema está optimizado para recomendaciones múltiples (varios inputs)
- Las similitudes se normalizan a escala 0-1 para mejor interpretabilidad
- Capacidad de exploración interactiva de features mediante gráficos Altair