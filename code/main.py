import functools
import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np # no usar de esta forma
from unidecode import unidecode # no usar de esta forma

import bq_process
import executor
# import gclogging
import properties
import task
from cluster_gen import ClusterGen

from sklearn.metrics import silhouette_score
from itertools import product
from scipy.spatial.distance import cdist
"""
All environ variables and general variables should be here
os library is already imported for your convenience
E.g.
"""
layer = os.environ.get("LAYER", "unset LAYER")
table = os.environ.get("TABLE", "unset TABLE")
TIME_FORMAT = "%Y-%m-%d %H:%M"

# logging configuration

job_uid = datetime.now().strftime("%Y%d%m_%H%M%S")
labels = {
    "type": "python-executor",
    "job_uid": job_uid,
    "process_uid": __name__,
    "layer": layer,
    "table": table,
}

# gclogging.nodefaults()
# logging.basicConfig(format=gclogging.FORMAT, level=logging.INFO)
# logger = gclogging.add_handler(logging.getLogger(__name__), __name__, labels, True)


def functionspy(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        global logger
        message = f"{func.__name__}({args}, {kwargs})"
        try:
            logger.debug(f"< {message}")
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("# %s raised an exception %s", func.__name__, str(e))
            raise e
        finally:
            logger.debug(f"> {message}")

    return decorator


def valid_args(*types):
    def check_args(f):
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwargs):
            for a, t in zip(args, types):
                assert isinstance(a, t), "arg %r does not match %s" % (a, t)
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_args


@functionspy
@valid_args(str)
def validation(query_file):
    validation_table = bq_process.bq_query(query_file).to_dataframe()
    if validation_table.flag.unique().all():
        logger.info("Ok")
        return True
    else:
        logger.error("Error")
        return False

## busqueda de mejores parametros
def random_search_params(df, eps_range, min_samples_range, num_iterations, metric, seed):
    if seed is not None:
        np.random.seed(seed)

    best_params = None
    best_silhouette = float('-inf')  

    for _ in range(num_iterations):
        eps_in_km = np.random.uniform(eps_range[0], eps_range[1])
        min_samples = np.random.randint(min_samples_range[0], min_samples_range[1] + 1)
        selected_metric = np.random.choice(metric)  # Cambiamos el nombre de la variable a selected_metric

        clustered_df = ClusterGen().cluster_passengers(df, eps_in_km, min_samples, selected_metric)

        silhouette = silhouette_score(df[["lat", "lon"]], clustered_df['zone'])
        
        print(eps_in_km,min_samples,selected_metric,silhouette)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_params = {'eps_in_km': eps_in_km, 'min_samples': min_samples, 'metric': selected_metric}  # Usamos selected_metric aquí
    print(best_params)
    return best_params

def grid_search_params(df, n_clusters_range, num_iterations, seed):
    if seed is not None:
        np.random.seed(seed)

    best_params = None
    best_silhouette = float('-inf')

    for n_clusters in n_clusters_range:
        silhouette_scores = []
        for _ in range(num_iterations):
            clustered_df = ClusterGen().cluster_passengers(df, n_clusters)
            silhouette = silhouette_score(df[["lat", "lon","lat_com","lon_com","lat_ae","lon_ae","dis_com","dis_ae"]], clustered_df['zone'])
            silhouette_scores.append(silhouette)

        print(n_clusters,silhouette)

        avg_silhouette = np.mean(silhouette_scores)
        if avg_silhouette > best_silhouette:
            best_silhouette = avg_silhouette
            best_params = {'n_clusters': n_clusters, 'avg_silhouette': avg_silhouette}

    print(best_params)
    return best_params


## borrar esta función si no es necesario
def clean_txt(txt):
    txt = txt.replace('ñ', 'n')
    txt = txt.replace('Ñ', 'N')
    return unidecode(txt)


# Calcula la matriz de distancias utilizando la fórmula de Haversine
import numpy as np

# Define la función para calcular la distancia de Haversine entre dos pares de coordenadas
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radio de la Tierra en kilómetros
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def calcular_cantidades(distancias, k):
    cantidades = np.where(distancias > k, 0, 1)
    cant_h = np.sum(cantidades, axis=1)  # Suma de cada fila (horizontalmente)
    cant_v = np.sum(cantidades, axis=0)  # Suma de cada columna (verticalmente)
    cant_vecinos = cant_v + cant_h
    return cant_vecinos

def main():
    # df = bq_process.bq_query(
    #     f"""SELECT 
    #     BP as bp, 
    #     latitud_obs as lat_obs, 
    #     longitud_obs as lng_obs, 
    #     direccion, 
    #     comuna,
    #     lat as lat,
    #     lon as lng
    #     FROM {properties.PROJECT_CONFIG}.data.base_oficial WHERE lat is not NULL""",
    #     cmd=True,
    # ).to_dataframe()
    print("Current Working Directory:", os.getcwd())
    df = pd.read_excel('C:/Repos/PoC - Vans Optimizer/code/base2.xlsx', sheet_name='base_oficial')
    for columna in df.select_dtypes(include=[object]):  # Select object type column
        df[columna] = df[columna].apply(clean_txt)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lon"], errors="coerce")

## agregar las informacion de comunas y zonas
    csv_comunas = "C:/Repos/PoC - Vans Optimizer/code/latlon-chile.csv"
    df_comunas = pd.read_csv(csv_comunas)
    df_comunas = df_comunas.rename(columns={"Comuna": "comuna"})
    df_comunas['comuna'] = df_comunas['comuna'].str.upper()

    df_comunas = df_comunas.rename(columns={df_comunas.columns[0]: 'CUT'})
    
    df_comunas = df_comunas.rename(columns={df_comunas.columns[8]: 'lat_com'})
    df_comunas = df_comunas.rename(columns={df_comunas.columns[9]: 'lon_com'})

    num_comunas = df_comunas["comuna"].nunique()

    df = pd.merge(df, df_comunas[['comuna', 'CUT', 'lat_com', 'lon_com']], on='comuna', how='inner')
    # filtra zona metropolitana
    cut_values = [13101, 13102, 13103, 13104, 13105, 13106, 13107, 13108, 13109, 13110, 13111, 13112, 13113, 13114,
                  13115, 13116, 13117, 13118, 13119, 13120, 13121, 13122, 13123, 13124, 13125, 13126, 13127, 13128,
                  13129, 13130, 13131, 13132, 13201, 13202, 13203, 13301, 13302, 13303, 13401, 13402, 13403, 13404,
                  13501, 13502, 13503, 13504, 13505, 13601, 13602, 13603, 13604, 13605]
    df = df[df['CUT'].isin(cut_values)]
    df = df.drop('CUT', axis=1)

    df= df[(df['lat'] >= -35) & (df['lat'] <= -30)] #punto de brasil

    df['lat_ae'] = -33.396926
    df['lon_ae'] = -70.7984857

    df['dis_com'] = 6371 * np.arccos(np.cos(np.radians(90 - df['lat'])) * np.cos(np.radians(90 - df['lat_com'])) + np.sin(np.radians(90 - df['lat'])) * np.sin(np.radians(90 - df['lat_com'])) * np.cos(np.radians(df['lon'] - df['lon_com'])))

    df['dis_ae'] = 6371 * np.arccos(np.cos(np.radians(90 - df['lat'])) * np.cos(np.radians(90 - df['lat_ae'])) + np.sin(np.radians(90 - df['lat'])) * np.sin(np.radians(90 - df['lat_ae'])) * np.cos(np.radians(df['lon'] - df['lon_ae'])))
#########################################################################

    latitudes = df['lat'].values
    longitudes = df['lon'].values

    # Calcula la matriz de distancias
    num_puntos = len(df)
    distancias = np.zeros((num_puntos, num_puntos))

    for i in range(num_puntos):
        for j in range(num_puntos):
            distancias[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
    
    #print(distancias)

######      ##############    ############# ########## ############
    k = 20
    vecinos = calcular_cantidades(distancias, k)
    df = df.assign(vecinos= vecinos)



## obtencion de los parametros
    best_params = grid_search_params(df, n_clusters_range=range(5, 20), num_iterations=10, seed=42) #kmeans
##

    c = ClusterGen()
    # df = df.head(50)

    df = c.cluster_passengers(df, best_params['n_clusters'])

    # # Calling the graph_zones
    
    c.graph_zones(df, "")
    
    #logger.info("Map has been generated and saved as 'cluster_map.html'.")

if __name__ == "__main__":
    # GOOGLE_APPLICATION_CREDENTIALS=credential.json
    # logger.info("Executing Main.py")
    # logger.info("Creating table control if not exists")
    # executor.ddl_executor(task.tabla_control)
    try:
        main()
        # bq_process.update_status("main", "Ok")
    except Exception as e:
        # bq_process.update_status("main", "Error")
        raise e
