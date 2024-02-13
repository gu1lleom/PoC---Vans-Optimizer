import pandas as pd
from utils import plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import numpy as np

def calculate_k_distance(coords, n_neighbors=10):
    """
    Calculate the k-distance to determine the value of eps.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='haversine')
    nbrs.fit(np.radians(coords))
    distances, _ = nbrs.kneighbors(np.radians(coords))
    k_dist = distances[:, -1]
    k_dist.sort()
    return k_dist

def find_elbow_point(k_dist):
    """
    Plot the k-distance graph to visually find the elbow point.
    """
    plt.plot(k_dist)
    plt.ylabel('k-distance')
    plt.xlabel('Points sorted by distance')
    plt.show()

def cluster_passengers(df, eps_in_km, min_samples):
    """
    DBSCAN to cluster passengers based on their geographic location.
    The eps parameter is the distance in kilometers converted to radians.
    """
    # Geographic coordinates to radians for haversine distance
    coords = df[['lat', 'lng']].to_numpy()
    
    # Earth's radius in kilometers
    earth_radius_in_km = 6371.0088  
    # eps to radians for DBSCAN
    eps_in_radians = eps_in_km / earth_radius_in_km

    # DBSCAN with haversine metric
    dbscan = DBSCAN(eps=eps_in_radians, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
    
    # Assign cluster labels to the original dataframe
    df['zone'] = dbscan.labels_
    
    return df


def segment_large_clusters_hierarchical(df, max_size=200):
    large_clusters = df['zone'].value_counts()[df['zone'].value_counts() > max_size].index
    for cluster_id in large_clusters:
        # Filter points belonging to the large cluster
        cluster_points = df[df['zone'] == cluster_id][['lat', 'lng']]
        # Estimate the number of subclusters needed
        n_subclusters = int(np.ceil(len(cluster_points) / max_size))
        agglomerative = AgglomerativeClustering(n_clusters=n_subclusters, linkage='ward')
        subcluster_labels = agglomerative.fit_predict(cluster_points)
        
        # Assign new subcluster labels
        df.loc[df['zone'] == cluster_id, 'subzone'] = subcluster_labels
    return df



df = pd.read_excel(r'C:\Users\Acid Labs\Desktop\Project Latam\02-04\pro2 (2) (1)\pro2 (2)\pro2\acid_labs-poc-vans-optimizer-4f730ee3cc6b\algorithm\algorithm\base.xlsx') 
coords = df[['lat', 'lng']].to_numpy()
# Calculate k-distance for all points
k_distances = calculate_k_distance(coords, n_neighbors=10)
    
# k-distances Plot to visually find the elbow point
find_elbow_point(k_distances)
earth_radius_km = 6371.0088
elbow_index = 100  
eps_km = k_distances[elbow_index] * earth_radius_km

min_samples_value = max(2, int(len(df) * 0.01))  # min_samples should be at least 2

   
# DBSCAN clustering with the chosen eps and min_samples
df = cluster_passengers(df, eps_in_km=0.8, min_samples=10) 
df = segment_large_clusters_hierarchical(df, max_size=200)
print(f"Total number of zones: {df['zone'].nunique()}")
print(df['zone'].value_counts())

