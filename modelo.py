import pandas as pd
import numpy as np
import gdown
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from sklearn.metrics.pairwise import cosine_distances

# Descarga y carga del dataset
@st.cache_data
def load_dataset():
    url = "https://drive.google.com/uc?id=1ppG70SIjTax3_zz6Nyobtbfm_4Xw-IzL"
    output = "peliculas.csv"
    gdown.download(url, output, quiet=True)
    df = pd.read_csv(output, sep=";")
    return df


# -----------------------------
# Clases de preprocesamiento
# -----------------------------
class TextContentCleaner(BaseEstimator, TransformerMixin):
    """
    Limpia los valores de texto separados por coma: hace un split por coma, convierte a minúsculas, elimina separadores, y devuelve una lista.
    """
    def __init__(self, features):
        self.features = features

    def clean_data(self, text_list):
        """Elimina espacios, convierte a minúsculas, y devuelve el string como lista, separados por la coma"""
        if isinstance(text_list, str):
            # Divide la cadena, elimina espacios, convierte a minúsculas
            return [str.lower(i.replace(" ", "")) for i in text_list.split(',')]
        # Si ya está vacío (por el transformer anterior), devuelve lista vacía
        return []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for feature in self.features:
            # Aplica la función de limpieza
            X_copy[feature] = X_copy[feature].apply(self.clean_data)
        return X_copy

# -------------------------------
# Transformador de texto (TF-IDF)
# -------------------------------
class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self
    def transform(self, X):
        return self.vectorizer.transform(X)


class ColumnDropper(BaseEstimator, TransformerMixin):
  """Transformador para eliminar columnas específicas."""

  def __init__(self, columns_to_drop):
    self.columns_to_drop = columns_to_drop

  def fit(self, X, y=None):
    return self

  def transform(self, X):
  # Devuelve una copia del DataFrame sin las columnas especificadas
    return X.drop(columns=self.columns_to_drop, errors='ignore')

class SoupCreator(BaseEstimator, TransformerMixin):
    """
    Crea una 'sopa de palabras' combinando múltiples columnas de texto.
    Permite indicar:
      - qué columnas mezclar,
      - separador personalizado.
    """
    def __init__(self, columns=None, separator=" "):
        """
        columns: list
          lista de columnas que deben unirs
        separator: str
            Separador entre palabras al unir las columnas.
        """
        self.columns = columns or []
        self.separator = separator

    def _combine_row(self, row):
        parts = []
        for col in self.columns:
            value = row[col] if col in row else ""
            if isinstance(value, list):
                joined = self.separator.join(map(str, value))
            elif isinstance(value, str):
                joined = value.replace(",", self.separator)
            else:
                joined = ""
            parts.append(joined)
        return self.separator.join(filter(None, parts))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        print(X.info())
        X_copy["soup"] = X_copy.apply(self._combine_row, axis=1)
        return X_copy


# Entrenamiento del modelo
@st.cache_resource
def train_model():
    df = load_dataset()

    text_list_cols = ["genres", "main_actors", "keywords", "directors"]
    columnas_no_usadas = ["runtime", "poster_path", "popularity", "vote_average", "budget", "revenue"]
    soup_columns = [
        "genres",
        "main_actors",
        "keywords",
        "directors",
        "overview",
        "production_companies",
        "title",
    ]

    preprocessor = Pipeline(
        [
            ("drop_cols", ColumnDropper(columnas_no_usadas)),
            ("clean_text", TextContentCleaner(features=text_list_cols)),
            ("soup", SoupCreator(soup_columns)),
        ]
    )

    X = preprocessor.fit_transform(df)
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(X["soup"])

    nn = NearestNeighbors(
        n_neighbors=11, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(tfidf_matrix)

    return df, X, tfidf, tfidf_matrix, nn


def recomendar_peliculas_multiples(idx_peliculas, n=10, pesos=None, usar_mediana=False, explicar=False):
    """
    Recomendar películas basadas en múltiples películas base.
    
    Args:
        idx_peliculas: Lista de índices de películas base
        n: Número de recomendaciones a devolver
        pesos: Lista de pesos para cada película base (default: pesos iguales)
        usar_mediana: Si True, usa la mediana de distancias en lugar de la suma
        explicar: Si True, incluye explicación de características compartidas
    """
    df, X, tfidf, tfidf_matrix, nn = train_model()

    if isinstance(idx_peliculas, int):
        idx_peliculas = [idx_peliculas]

    idx_peliculas_validas = [
        idx for idx in idx_peliculas if 0 <= idx < len(df)
    ]

    if len(idx_peliculas_validas) == 0:
        return pd.DataFrame()  # Devolver DataFrame vacío si no hay películas válidas

    # Normalizar pesos si se proporcionan, o usar pesos iguales
    if pesos is None:
        pesos = np.ones(len(idx_peliculas_validas)) / len(idx_peliculas_validas)
    else:
        pesos = np.array(pesos[:len(idx_peliculas_validas)])
        pesos = pesos / pesos.sum()  # Normalizar para que sumen 1

    # Matriz para guardar todas las distancias (películas × bases)
    todas_distancias = np.zeros((tfidf_matrix.shape[0], len(idx_peliculas_validas)))

    # Calcular distancias para cada película base
    for i, idx in enumerate(idx_peliculas_validas):
        distancias = cosine_distances(tfidf_matrix[idx], tfidf_matrix)[0]
        todas_distancias[:, i] = distancias

    # Combinar distancias según el método elegido
    if usar_mediana:
        distancias_totales = np.median(todas_distancias, axis=1)
    else:
        # Suma ponderada de distancias
        distancias_totales = np.sum(todas_distancias * pesos.reshape(1, -1), axis=1)

    # Excluir películas base
    distancias_totales[idx_peliculas_validas] = np.inf

    # Obtener mejores matches
    mejores_indices = np.argsort(distancias_totales)[:n]
    
    # Normalizar similitudes a 0-1
    if usar_mediana:
        similitudes = 1 - distancias_totales[mejores_indices]
    else:
        similitudes = 1 - (distancias_totales[mejores_indices] / len(idx_peliculas_validas))

    # Crear DataFrame con recomendaciones
    df_recomendadas = df.iloc[mejores_indices].copy()
    df_recomendadas['similitud_promedio'] = similitudes

    # Añadir contribuciones individuales de cada película base
    for i, idx_base in enumerate(idx_peliculas_validas):
        nombre_col = f'similitud_con_{df.loc[idx_base, "title"]}'
        df_recomendadas[nombre_col] = 1 - todas_distancias[mejores_indices, i]

    if explicar:
        # Extraer y comparar features importantes
        caracteristicas = []

        # Nombres de features TF-IDF
        try:
            feature_names = tfidf.get_feature_names_out()
        except Exception:
            feature_names = None

        for idx_rec in mejores_indices:
            try:
                # Helpers para obtener conjuntos seguros desde columnas separadas por comas
                def split_set(col, idx):
                    val = df.loc[idx, col]
                    if pd.isna(val):
                        return set()
                    # Convertir a string y split, strip
                    return set([s.strip() for s in str(val).split(',') if s is not None and str(s).strip() != ""])

                generos_rec = split_set('genres', idx_rec)
                directores_rec = split_set('directors', idx_rec)
                actores_rec = split_set('main_actors', idx_rec)
                keywords_rec = split_set('keywords', idx_rec)
                prod_rec = split_set('production_companies', idx_rec)

                generos_comunes = {}
                keywords_comunes = {}
                prod_comunes = {}

                for idx_base in idx_peliculas_validas:
                    generos_base = split_set('genres', idx_base)
                    keywords_base = split_set('keywords', idx_base)
                    prod_base = split_set('production_companies', idx_base)

                    base_title = df.loc[idx_base, 'title']
                    generos_comunes[base_title] = sorted(list(generos_rec & generos_base))
                    keywords_comunes[base_title] = sorted(list(keywords_rec & keywords_base))
                    prod_comunes[base_title] = sorted(list(prod_rec & prod_base))

                # Directores y actores en común (si aparecen en cualquiera de las bases)
                directores_comunes = [d for d in sorted(directores_rec) if any(d in str(df.loc[idx_base, 'directors']) for idx_base in idx_peliculas_validas)]
                actores_comunes = [a for a in sorted(actores_rec) if any(a in str(df.loc[idx_base, 'main_actors']) for idx_base in idx_peliculas_validas)]

                # Top TF-IDF terms que explican la similitud: usar producto elemento a elemento entre vectores
                top_terms = []
                try:
                    if feature_names is not None:
                        vec_rec = tfidf_matrix[idx_rec]
                        # acumulador de scores (sum of pairwise products with each base, ponderado por pesos si aplica)
                        prod_scores = np.zeros(len(feature_names))
                        for i, idx_base in enumerate(idx_peliculas_validas):
                            vec_base = tfidf_matrix[idx_base]
                            pr = vec_rec.multiply(vec_base).toarray().ravel()
                            w = (pesos[i] if (pesos is not None and not usar_mediana) else 1.0)
                            prod_scores += pr * float(w)
                        # tomar top k términos con mayor score
                        top_k = 8
                        if prod_scores.sum() > 0:
                            top_idx = np.argsort(prod_scores)[::-1][:top_k]
                            for j in top_idx:
                                if prod_scores[j] > 0:
                                    top_terms.append(str(feature_names[j]))
                        # evitar duplicados y truncar
                        top_terms = list(dict.fromkeys(top_terms))[:top_k]
                except Exception:
                    top_terms = []

                # Contribuciones por base: usar similitud individual (1 - distancia)
                sim_per_base = []
                for i, idx_base in enumerate(idx_peliculas_validas):
                    try:
                        sim = 1.0 - float(todas_distancias[idx_rec, i])
                    except Exception:
                        sim = 0.0
                    # clamp
                    if sim < 0:
                        sim = 0.0
                    sim_per_base.append(sim)

                sim_array = np.array(sim_per_base, dtype=float)
                if usar_mediana:
                    contrib_raw = sim_array.copy()
                else:
                    peso_array = np.array(pesos[:len(sim_array)]) if pesos is not None else np.ones_like(sim_array)
                    contrib_raw = sim_array * peso_array

                if contrib_raw.sum() > 0:
                    contribs = (contrib_raw / contrib_raw.sum()).tolist()
                else:
                    contribs = [0.0] * len(contrib_raw)

                contrib_dict = {df.loc[idx_base, 'title']: float(contribs[i]) for i, idx_base in enumerate(idx_peliculas_validas)}

                elementos_comunes = {
                    'generos': generos_comunes,
                    'keywords': keywords_comunes,
                    'production_companies': prod_comunes,
                    'directores': directores_comunes,
                    'actores': actores_comunes,
                    'top_terms': top_terms,
                    'contribuciones': contrib_dict,
                    'similitudes_individuales': {df.loc[idx_base, 'title']: float(1.0 - float(todas_distancias[idx_rec, i])) for i, idx_base in enumerate(idx_peliculas_validas)}
                }
            except Exception:
                # Si hay algún error, crear un diccionario vacío pero válido
                elementos_comunes = {
                    'generos': {},
                    'keywords': {},
                    'production_companies': {},
                    'directores': [],
                    'actores': [],
                    'top_terms': [],
                    'contribuciones': {},
                    'similitudes_individuales': {}
                }
            caracteristicas.append(elementos_comunes)
        
        df_recomendadas['explicacion'] = caracteristicas
    
    # Guardar el índice original del DataFrame `df` para poder mapear los resultados
    # al conjunto original cuando se necesite (por ejemplo para obtener el perfil TF-IDF).
    # `mejores_indices` viene como numpy array; aseguramos longitud correcta.
    df_recomendadas = df_recomendadas.reset_index(drop=True)
    df_recomendadas['orig_index'] = list(mejores_indices)

    return df_recomendadas

