import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'k-means')))
import kmeans as kmeans


def DaviesBouldin(X, centroids, clusters):
    n_cluster = len(centroids)
    cluster_k = [X[clusters == k] for k in range(n_cluster)]
    variances = [np.mean([np.linalg.norm(p - centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = np.empty(shape=(n_cluster, n_cluster))

    for i in range(n_cluster):
        for j in range(n_cluster):
            if i != j:
                db[i][j] = (variances[i] + variances[j]) / np.linalg.norm(centroids[i] - centroids[j])

    for i in range(n_cluster):
        db[i][i] = -1

    return np.sum(np.max(db, axis=1)) / n_cluster


def CHIndex(X, centroids, clusters):
    k = len(centroids)
    mean = np.mean(centroids, axis=0)
    W = np.sum([(x - centroids[clusters[i]])**2 for i, x in enumerate(X)])
    B = np.sum([len(X[clusters == i]) * (c - mean)**2 for i, c in enumerate(centroids)])
    n = len(X)
    return (n - k) * B / ((k - 1) * W)


def inner_criteria(X, k_range, iterations=200):
    DB = dict()  # Davies-Bouldin
    CH = dict()  # Calinski-Harabasz

    for k in k_range:
        if k < 2:
            continue
        centroids, clusters = kmeans.k_means(X, k, iterations)
        DB[k] = DaviesBouldin(X, centroids, clusters)
        CH[k] = CHIndex(X, centroids, clusters)

    ch_list = list(CH.items())

    db_best = min(DB, key=DB.get)
    ch_best = 0
    delta = sys.maxsize
    for k in range(1, len(CH) - 1):
        temp = ch_list[k + 1][1] - 2 * ch_list[k][1] + ch_list[k - 1][1]
        if temp < delta:
            delta = temp
            ch_best = ch_list[k][0]

    return DB, db_best, CH, ch_best


def outer_criteria(X, k_range, reference, iterations=200):
    n = len(X)
    RS = dict()  # Rand Statistic
    FM = dict()  # Fowlkes-Mallows

    for k in k_range:
        TP, FN, FP, TN = (0,)*4
        if k < 2:
            continue

        _, clusters = kmeans.k_means(X, k, iterations)
        # Compute TP, FN, FP, TN.
        for i in range(n):
            for j in range(i + 1, n):
                if clusters[i] == clusters[j] and reference[i] == reference[j]:
                    TP += 1
                elif clusters[i] != clusters[j] and reference[i] == reference[j]:
                    FN += 1
                elif clusters[i] == clusters[j] and reference[i] != reference[j]:
                    FP += 1
                else:
                    TN += 1
        RS[k] = (TP + TN) / n
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        FM[k] = np.sqrt(precision * recall)

    best_rs = max(RS, key=RS.get)
    best_fm = max(FM, key=FM.get)

    return RS, best_rs, FM, best_fm


if __name__ == '__main__':
    data_dir = 'data'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    clusters_range = range(2, 10)

    # Inner criterias' block.
    image_path = os.path.join(data_dir, 'policemen.jpg')
    image = np.array(Image.open(image_path), dtype=np.uint8)
    new_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    db, best_db, ch, best_ch = inner_criteria(new_image, clusters_range)
    best_inner = (best_db + best_ch) // 2

    # Save the clustered image.
    centroids, clusters = kmeans.k_means(new_image, best_inner, iterations=200)
    new_image = np.vstack([centroids[i] for i in clusters]).astype(np.uint8).reshape(image.shape)
    Image.fromarray(new_image).save(os.path.join(results_dir, '%d-clusters-policemen.jpg' % best_inner))

    # Outer criterias' block.
    outer_criterias_input = os.path.join(data_dir, 'outer_criterias_input.txt')
    data = np.loadtxt(outer_criterias_input, delimiter=' ')
    reference, points = data[:, 0], data[:, 1:]
    rs, best_rs, fm, best_fm = outer_criteria(points, clusters_range, reference)

    # Draw the results.
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax1, ax2, ax3, ax4 = ax.flatten()

    x_axes = list(clusters_range)

    ax1.scatter(x=x_axes, y=list(db.values()))
    ax1.set_title('Davies-Bouldin. Optimal $k$ is %d' % best_db)

    ax2.scatter(x=x_axes, y=list(ch.values()))
    ax2.set_title('Calinski-Harabasz. Optimal $k$ is %d' % best_ch)

    ax3.scatter(x=x_axes, y=list(rs.values()))
    ax3.set_title('Rand Statistic. Optimal $k$ is %d' % best_rs)

    ax4.scatter(x=x_axes, y=list(fm.values()))
    ax4.set_title('Fowlkes-Mallows. Optimal $k$ is %d' % best_fm)

    plt.show()
