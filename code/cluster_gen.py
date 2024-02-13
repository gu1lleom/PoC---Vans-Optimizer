import folium
import matplotlib
import numpy as np
from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

class ClusterGen:

    def cluster_passengers(self, df, n_clusters):
        print("Get Clusters")
        """
        K-Means Clustering to cluster passengers based on their geographic location.
        """
        coords = df[["lat", "lon","dis_com","dis_ae","vecinos"]].to_numpy()

        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)

        # Assign cluster labels to the original dataframe
        df["zone"] = kmeans.labels_

        return df


    def graph_zones(self, df, centroids):
        """
        Create a Folium map visualizing the clusters and their centroids.
        Each cluster (zone) have a unique color.
        """
        # Map centered around the average location
        map_center = [df["lat"].mean(), df["lon"].mean()]
        m = folium.Map(location=map_center, zoom_start=10)

        # Color palette with enough colors for each cluster using the new method
        num_clusters = df["zone"].nunique()
        cmap = matplotlib.colormaps["nipy_spectral"](
            np.linspace(0, 1, num_clusters)
        )  # Using the new access method

        # Color assignation for each zone
        zone_colors = {
            zone: matplotlib.colors.rgb2hex(cmap[i])
            for i, zone in enumerate(sorted(df["zone"].unique()))
        }

        # Each passenger point with the color based on its cluster
        for _, row in df.iterrows():
            cluster_index = row["zone"]
            if cluster_index != -1:
                color = zone_colors[cluster_index]
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.5,
                ).add_to(m)

        # Plot centroids with a distinctive feature
        # for index, row in centroids.iterrows():
        #     color = zone_colors[row['zone']] if row['zone'] in zone_colors else 'black'
        #     folium.CircleMarker(
        #         location=[row['lat'], row['lng']],
        #         radius=8,
        #         color='black',  # Black border for all centroids
        #         fill=True,
        #         fill_color=color,  # Fill color unique to each zone
        #         fill_opacity=0.9,
        #         popup=f'Centroid for Zone {row["zone"]}'
        #     ).add_to(m)

        # Save the map

        m.save("cluster_map.html")
        print("LISTO")
