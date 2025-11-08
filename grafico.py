from modelo import train_model
import pandas as pd
import numpy as np
from scipy.sparse import issparse

def obtener_perfil_contenido(idx_pelicula, n_features=15):
    """
    Devuelve las N features más importantes de una película.
    """
    df, X, tfidf, tfidf_matrix, nn = train_model()
    if issparse(tfidf_matrix):
        feature_vector = tfidf_matrix[idx_pelicula].toarray().flatten()
    else:
        feature_vector = tfidf_matrix[idx_pelicula].flatten()
    
    top_indices = np.argsort(feature_vector)[-n_features:][::-1]
    feature_names = np.array(tfidf.get_feature_names_out())
    top_features = feature_names[top_indices]
    top_scores = feature_vector[top_indices]

    df_features = pd.DataFrame({
        'feature': top_features,
        'peso_tfidf': top_scores
    })

    df_features = df_features[df_features['peso_tfidf'] > 0]
    return df_features


def generar_df_comparacion(peliculas_base_idx, n_recomendaciones=10, n_features=15):
    """
    Genera un DataFrame concatenado con los perfiles TF-IDF de todas las películas base y sus recomendaciones.
    """
    df, X, tfidf, tfidf_matrix, nn = train_model()
    perfiles_list = []

    for j, idx_base in enumerate(peliculas_base_idx):
        titulo_base = df.loc[idx_base, "title"]

        distances, indices = nn.kneighbors(
            tfidf_matrix[idx_base],
            n_neighbors=n_recomendaciones + 1
        )
        all_indices = indices.flatten()

        for i, idx in enumerate(all_indices):
            titulo = df.loc[idx, 'title']
            df_perfil = obtener_perfil_contenido(idx, n_features)

            if i == 0:
                grupo = f"Base {j+1}: {titulo_base}"
            else:
                grupo = f"Base {j+1} - Recom {i}: {titulo}"

            df_perfil['comparison_group'] = grupo
            df_perfil['base_index'] = idx_base
            perfiles_list.append(df_perfil)

    df_full_comparison = pd.concat(perfiles_list, ignore_index=True)
    return df_full_comparison