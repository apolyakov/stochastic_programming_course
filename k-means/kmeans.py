from PIL import Image
import numpy as np

import os
from copy import deepcopy


def k_means(points, k, iterations):
    n = points.shape[0]
    centroids = points[np.random.randint(n, size=k),:]
    new_centroids = np.zeros(centroids.shape)
    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    changed = True
    while changed and iterations:
        for i in range(k):
            distances[:, i] = np.linalg.norm(points - centroids[i], axis=1)

        clusters = np.argmin(distances, axis=1)

        for i in range(k):
            new_centroids[i] = np.mean(points[clusters == i], axis=0)

        if np.linalg.norm(centroids - new_centroids) == 0:
            changed = False

        centroids = deepcopy(new_centroids)
        iterations -= 1

    return centroids, clusters


if __name__ == '__main__':
    data_dir = 'data'
    files = os.listdir(data_dir)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    clusters_count = [2, 4, 8]
    iterations_count = 200

    for file in files:
        image = np.array(Image.open(os.path.join(data_dir, file)), dtype=np.uint8)
        points = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
        for num_clusters in clusters_count:
            for i in range(3):
                centroids, clusters = k_means(points, num_clusters, iterations_count)
                new_image = np.vstack(
                    [centroids[i] for i in clusters]).astype(np.uint8).reshape(image.shape)
                Image.fromarray(new_image).save(
                    os.path.join(results_dir, '%d-try-%d-clusters-%s' % (i, num_clusters, file)))

