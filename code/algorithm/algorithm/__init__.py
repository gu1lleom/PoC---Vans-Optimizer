from utils import pd, math, plt, np, folium, requests, json, DBSCAN, pdist, squareform, matplotlib, KMeans
from haversine import haversine, Unit
from sklearn.neighbors import NearestNeighbors
from filetreatment import FileTreatment
from optimizer import get_solution


# This distance can be executed once, then save the distances
# Passengers doesn't change so frequently their addresses
def get_airport_distance_google(df, coordenadas_aeropuerto, api_key):
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json?"
    
    # Convertir el DataFrame en una lista de coordenadas
    coordenadas = df.apply(lambda row: f"{row['lat']},{row['lng']}", axis=1).tolist()
    
    # Create an empty list to save distances
    distancias_a_aeropuerto = []
    
    for origen in coordenadas:
        parametros = {
            "origins": origen,
            "destinations": coordenadas_aeropuerto,  # airport 
            "key": api_key
        }
        
        respuesta = requests.get(base_url, params=parametros)
        data = respuesta.json()

        # Print the structure of the API response 
        print("API Response Structure:", json.dumps(data, indent=2))
        
        if data['status'] == 'OK':
            # Extract the distance value correctly based on the API response structure
            distancia = data['rows'][0]['elements'][0].get('distance', {}).get('value', None)
            
            if distancia is not None:
                distancia = distancia / 1000
            distancias_a_aeropuerto.append(distancia)
           # distancia = data['rows'][0]['elements'][0]['distance']['value']/1000
            #distancias_a_aeropuerto.append(distancia)
        else:
            print(f"Error en la solicitud para el origen {origen}")
            distancias_a_aeropuerto.append(None)  # Add null value in case of error

        # Update the 'distance' column in the DataFrame with the Google Maps API distances
    df['distance'] = distancias_a_aeropuerto

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two geographic points.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Radius of the Earth in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance


# Dynamically calculate an initial eps
def calculate_initial_eps(coords):
    
#Calculating an initial epsilon value based on the average distance to the nearest neighbor.

   # Ensure coords are in radians for Haversine distance
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric='haversine')
    nbrs.fit(np.radians(coords))
    distances, indices = nbrs.kneighbors(np.radians(coords))
    eps = np.median(distances[:, 1]) * 6371  # Converting median distance from radians to km
    return eps

def find_optimal_dbscan_params(df, initial_eps=None, min_samples_start=2, min_samples_end=5):
    if initial_eps is None:
        initial_eps = calculate_initial_eps(df[['lat', 'lng']].values)
    eps_range = np.linspace(initial_eps * 0.9, initial_eps * 1.1, 5)
    min_samples_range = range(min_samples_start, min_samples_end + 1)

    # Initializing 'num_clusters' with float('inf') instead of None
    best_configuration = {'eps': initial_eps, 'min_samples': min_samples_start, 'noise': len(df), 'num_clusters': float('inf')}
    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps / 6371.0, min_samples=min_samples, metric='haversine').fit(np.radians(df[['lat', 'lng']].values))
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Checking if current configuration is better based on the number of noise points and clusters
            if n_noise < best_configuration['noise'] and n_clusters < best_configuration['num_clusters']:
                best_configuration.update({'eps': eps, 'min_samples': min_samples, 'noise': n_noise, 'num_clusters': n_clusters})

    return best_configuration['eps'], best_configuration['min_samples']

# Defining functions for distance calculation
def cluster_passengers(df, eps, min_samples):
    db = DBSCAN(eps=eps / 6371.0, min_samples=min_samples, metric='haversine').fit(np.radians(df[['lat', 'lng']].values))
    df['cluster'] = db.labels_
    return df


def calculate_centroids(df):
    """
    Calculating centroids for each cluster, excluding noise points.
    Only 'lat' and 'lng' columns should be used to calculate the mean.
    """
    # Ensuring that we only including rows where 'zone' is >= 0 to exclude noise points
    # Then, grouping by 'zone' and calculating the mean only for 'lat' and 'lng' columns
    clusters = df[df.cluster != -1]
     # list ['lat', 'lng'] instead of a tuple ('lat', 'lng')
    centroids = clusters.groupby('cluster')[['lat', 'lng']].mean().reset_index()
    return centroids
# calculating the centroid of each zone
def calculate_zone_centroids(df, cluster_centers):
    centroids = pd.DataFrame(cluster_centers, columns=['lat', 'lng'])
    return centroids

# creating a distance matrix Haversine distances between points.
def create_distance_matrix(df, centroids=None, threshold=8):
    """
    Creating a symmetric matrix of Haversine distances between points.
    """
    coords = df[['lat', 'lng']].to_numpy()
    dist_matrix = squareform(pdist(coords, lambda u, v: haversine_distance(u, v)))

    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

# Calculating the distance to the centroid of each zone and add it to 'df'
def calculate_distance_to_centroid(row, centroids):
        zone = row['cluster']
        centroid = centroids.iloc[zone]
        return haversine_distance((row['lat'], row['lng']), (centroid['lat'], centroid['lng']))

def graph_zones(df, centroids):
    map_center = [df['lat'].mean(), df['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=10)
    color_map = cm.rainbow(np.linspace(0, 1, len(centroids['cluster'].unique())))
    
    for _, row in df.iterrows():
        if row['cluster'] != -1:
            color = matplotlib.colors.rgb2hex(color_map[row['cluster']])
            folium.CircleMarker([row['lat'], row['lng']], radius=5, color=color, fill=True).add_to(m)
            
    for _, row in centroids.iterrows():
        folium.Marker([row['lat'], row['lng']], icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    
    m.save('cluster_map.html')

if __name__ == "__main__":
    print("Initial processes")
    print("--Reading file")

    df = pd.read_excel(r'C:\Users\Acid Labs\Desktop\Project Latam\02-06\pro2 (2) (1)\latam-vans\latam-vans\acid_labs-poc-vans-optimizer-4f730ee3cc6b\algorithm\algorithm\base.xlsx')  # Replace with your actual data source
    
    # Ensuring that 'lat' and 'lng' are numeric, coercing errors to NaN
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')

    # Drop rows where either 'lat' or 'lng' is NaN
    df.dropna(subset=['lat', 'lng', 'zona_comuna'], inplace=True)

    # Retrieving unique values in 'zona_comuna' for iteration.
    zona_comunas = df['zona_comuna'].unique()

    for zona_comuna in zona_comunas:
        print(f"Processing {zona_comuna}")
        # Filtering the dataset for the current 'zona_comuna'.
        df_zone = df[df['zona_comuna'] == zona_comuna]

    print("--Assigning variables")
    # distance: is the harvesine distance between the address and the airport
    distances_to_airport = df_zone['distance'].values
    cities = df_zone['addr_city'].values
    pos = df_zone[['lat', 'lng']]
    passengers_num = len(pos)

    # Airport Coordinates
    coordenadas_aeropuerto = "33.3973, 70.7937"

    # Calculating distances using the Google Maps API
    api_key = "AIzaSyCHok1Z9dxRiDbanQj6scLcFp4_8tPJiv0"
    get_airport_distance_google(df_zone, coordenadas_aeropuerto, api_key)
   
    # Converting 'lat' and 'lng' to numeric and handle 'None' in 'distance'
    df_zone['distance'] = pd.to_numeric(df_zone['distance'], errors='coerce').fillna(0)  
    
   # Applying the optimal parameters for DBSCAN
    optimal_eps, optimal_min_samples = find_optimal_dbscan_params(df_zone)
    # Applying DBSCAN clustering with the chosen eps and min_samples 
    df_zone = cluster_passengers(df_zone, optimal_eps, optimal_min_samples)

    # First few rows to verify clustering results
    print(df_zone.head())

    # Displaying the total number of zones for debugging
    print(f"Total number of clusters: {df_zone['cluster'].nunique() - (1 if -1 in df_zone['cluster'].unique() else 0)}")
    print(df_zone['cluster'].value_counts())

    # Calculating centroids for the clusters
    centroids = calculate_centroids(df_zone)

    # Checking if centroids are calculated correctly
    if not centroids.empty:
        print("Centroids calculated successfully.")
        print(centroids)
    else:
        print("No centroids found. Check your clustering results.")

    # Creating distance matrix 
    distance_matrix = create_distance_matrix(df_zone, centroids=None, threshold=8)
    distances_to_airport = df_zone['distance'].values
    matrix_cleaned = distance_matrix.to_numpy(dtype=float)
    best_solution, best_cost = get_solution(distance_matrix=matrix_cleaned, distances_to_airport=distances_to_airport) 

    graph_zones(df_zone, centroids)

    # Aggregating results, to perform further analysis, or save outputs for each zone.
    print(f"Completed processing for {zona_comuna}")

    print("All zones processed.")

    # Getting the solution from the optimizer, ensuring it can handle the data provided
    print("Getting solution...")
    best_solution, best_cost = get_solution(distance_matrix=distance_matrix.to_numpy(), distances_to_airport=distances_to_airport)

    print(df_zone['cluster'].nunique())
    print(df_zone['cluster'].value_counts())

    # Calling the graph_zones function with the necessary arguments
    graph_zones(df, centroids)
    print("Map has been generated and saved as 'cluster_map.html'.")

