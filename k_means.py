import numpy as np



def k_means(data, k):
    ''' data:   list of vectors (data points)
        k:      number of clusters
    '''
    cluster_centers = data[np.random.choice(data.shape[0], k, replace=False)]
    new_cluster_centers = np.zeros_like(cluster_centers)

    while True:
        # Assignment
        dist = np.zeros((data.shape[0], k))
        for i, center in enumerate(cluster_centers):
            dist[:, i] = np.linalg.norm(data - center, axis=1)

        clusters = np.argmin(dist, axis=1)


        # Update
        for i in range(k):
            new_cluster_centers[i] = np.mean(data[clusters == i], axis=0)


        if np.all(new_cluster_centers == cluster_centers):
            break

        cluster_centers = new_cluster_centers

    return clusters, cluster_centers




if __name__ == '__main__':
    from matplotlib import pyplot as plt

    np.random.seed(10)

    data = np.random.rand(20, 2)
    x, y = zip(*data)

    clusters, centers = k_means(data, k=3)

    colors = ['red', 'green', 'blue']
    plt.scatter(x, y, color=[colors[c] for c in clusters])

    c_x, c_y = zip(*centers)
    plt.scatter(c_x, c_y, marker='x')

    plt.show()
        