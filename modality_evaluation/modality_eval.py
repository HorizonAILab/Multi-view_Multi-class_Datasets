import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from utils import load_multi_view_data
from utils import plot_radar_chart
from tqdm import tqdm


# get all dataset name in the new_datasets folder and split the name by '.'
dataset_names = os.listdir('new_datasets')
dataset_names = [name.split('.')[0] for name in dataset_names]

basic_path = r"./new_datasets"
# dataset_name = "100Leaves"
metrics = ['NMI', 'Silhouette']

for dataset_name in tqdm(dataset_names):
    # Load data
    X, Y, num_classes = load_multi_view_data(
        basic_path=basic_path,
        dataset_name=dataset_name,
        normalization='Standard'
    )

    # K-means clustering
    views = []
    scores = {}
    for metric in metrics:
        scores[metric] = []
    for i, view_data in enumerate(X):
        kmeans = KMeans(n_clusters=num_classes, random_state=42)
        cluster_labels = kmeans.fit_predict(view_data)

        # Evaluation
        scores['Silhouette'].append(silhouette_score(view_data, cluster_labels))
        scores['NMI'].append(normalized_mutual_info_score(Y.flatten(), cluster_labels))
        views.append(f"View {i+1}")

    # Plot and save radar chart
    plot_radar_chart(views, dataset_name, scores, save_path='./radar_figs')