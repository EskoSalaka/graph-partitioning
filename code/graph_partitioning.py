import numpy as np
from scipy.sparse.csgraph import laplacian

from scipy.sparse.linalg import eigsh
from scipy.sparse import diags, csr_matrix, spdiags, eye
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


###################################################################
# Helper functions
###################################################################
def get_adjacency_matrix(file_path):
    """Returns a sparse and symnmetric the adjecancy matrix from a given file containing the graph in a format where one 
    line contains one edge in the shape of vertexid1 vertexid2. """
    with open(file_path) as f:
        num_vertices = int(f.readline().split()[2])

    data = np.genfromtxt(file_path, dtype=np.float)

    row = np.array(data[:, 0])
    col = np.array(data[:, 1])
    d = np.array(np.ones(len(row)))

    # This makes the matrix symmetric and also ignores self-links
    A = csr_matrix((d, (row, col)), shape=(num_vertices, num_vertices))
    A = A + A.T

    return A - diags(A.diagonal())


def get_laplacian(A, type="unnormalized"):
    """Returns a sparse Laplacian from a given adjecancy matrix. It can either be of type unnormalized,
    normalized or random-walk"""
    if type == "unnormalized":
        return laplacian(A, normed=False)
    elif type == "normalized":
        return laplacian(A, normed=True)
    elif type == "randomwalk":
        D_inv = spdiags(1.0 / A.sum(axis=1).flatten(), [0], A.shape[0], A.shape[0], format='csr')
        return eye(A.shape[0], A.shape[0]) - np.dot(D_inv, A)


def get_eigs(L, k, normed_eigenvector_rows=False):
    """Returns the first k eigenvectors and values from a give sparse Laplacian matrix. Optionally,
    the rows of the eigenvectors can be normalized. The eigs are calculated by using the scipy eigsh function
    which uses ARPACK."""
    vals, vecs = eigsh(L, k=k, which="SM")

    if normed_eigenvector_rows:
        vecs = normalize(vecs, axis=1)

    return vals, vecs


def eig_kmeans(U, k):
    """Returns the kmeans clustering of the given eigenvectors of a Laplacian matrix. It uses the first k eigenvectors.
    It return the original labellings of the sklearns kmeans function and also returns the partitions as a dictionary.
    The partitions are in the form {partition_id: [vertex_ids in partition_id]}
    """
    km = KMeans(n_clusters=k, n_jobs=8, init="k-means++", n_init=100, precompute_distances=True).fit(U[:, 0:k])

    partitions = {}

    for k in range(max(km.labels_) + 1):
        partitions[k] = np.where(km.labels_ == k)[0]

    return km.labels_, partitions


###################################################################
# The main partition function
###################################################################
def kmeans_partitions(k, laplacian_type="unnormalized", normed_eigenvector_rows=False, A=None, in_file=None, out_file=None):
    """This is the main partitioning function. It does the spectral partitioning for a given sparse adjecancy matrix for
    given k partitions. The laplacian_type can be specified as 'normalized', 'unnormalized' or 'randomwalk'. Lastly,
    the rows of the Laplacian can be optionally normalized by setting 'normed_eigenvector_rows' true.
    The partitions are returned as described in the 'eig_kmeans' function.

    Optionally, instead of an adjecancy matrix, a path to a file containing the graph can be given. If the 'out_file'
    is given, the partitions will be written in the given text file as instructed in the project description.
    The text file will be of the form vertexID clusterID on each line.

    Example run: kmeans_partitions(k=5,
                                   normed_eigenvector_rows=False,
                                   laplacian_type="normalized",
                                   in_file="../data/Oregon-1.txt",
                                   out_file="Oregon-1_partition.txt")
    """
    if in_file:
        A = get_adjacency_matrix(in_file)

    L = get_laplacian(A, type=laplacian_type)
    vals, vecs = get_eigs(L, k, normed_eigenvector_rows=normed_eigenvector_rows)

    labels, partitions = eig_kmeans(vecs, k)

    if out_file:
        np.savetxt(out_file, np.dstack((np.arange(0, labels.size), labels))[0], delimiter=' ', fmt="%d")

    return labels, partitions


###################################################################
# Other handy functions
###################################################################
def partition_sizes(labels):
    """Returns the size of each partition"""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def objective(partitions, A):
    """Returns the objective function (conductance) for partitions as instructed in the project description."""
    partition_num = len((partitions.keys()))
    obj = 0

    for k in range(partition_num):
        indices = partitions[k]
        other = np.setdiff1d(np.arange(0, A.shape[0]), indices)

        obj += np.sum(A[indices, :][:, other]) / len(indices)

    return obj


###################################################################
# A function that compares the random-walk and normalized Laplacians
###################################################################
def test(A, k_values):
    """Compares the random-walk and normalized Laplacians with and without normalizing the rows of the eigenvectors.
    It does the partitioning for a range of k_values and compares the objective function and the partition sizes for
    each type of Laplacian. Example of this function is shown in the Jupyter notebook 'Comparing Laplacians',
    """
    L_rw = get_laplacian(A, type="randomwalk")
    L_norm = get_laplacian(A, type="normalized")

    vals_rw, eigs_rw = get_eigs(L_rw, k=30, normed_eigenvector_rows=False)
    vals_norm, eigs_norm = get_eigs(L_norm, k=30, normed_eigenvector_rows=False)

    eigs_rw_r = normalize(eigs_rw, axis=1)
    eigs_norm_r = normalize(eigs_norm, axis=1)

    obj_rw = []
    obj_norm = []
    obj_rw_r = []
    obj_norm_r = []

    for k in k_values:
        km_rw = eig_kmeans(eigs_rw, k=k)
        km_norm = eig_kmeans(eigs_norm, k=k)
        km_rw_r = eig_kmeans(eigs_rw_r, k=k)
        km_norm_r = eig_kmeans(eigs_norm_r, k=k)

        obj_rw.append(objective(km_rw[1], A))
        obj_norm.append(objective(km_norm[1], A))
        obj_rw_r.append(objective(km_rw_r[1], A))
        obj_norm_r.append(objective(km_norm_r[1], A))

        print("================================")
        print(f"Partition sizes for k={k} partitions")
        print("================================")
        print()
        print("Normalized Laplacian")
        print(partition_sizes(km_norm[0]))
        print()
        print("Normalized Laplacian with normalized rows ")
        print(partition_sizes(km_norm_r[0]))
        print()
        print("Random-walk Laplacian")
        print(partition_sizes(km_rw[0]))
        print()
        print("Random-walk Laplacian with normalized rows")
        print(partition_sizes(km_rw_r[0]))
        print()

    plt.plot(k_values, obj_norm, color='green', label='Normalized Laplacian')
    plt.plot(k_values, obj_norm_r, color='green', linestyle='dashed', label='Normalized Laplacian and normalized rows')
    plt.plot(k_values, obj_rw, color='red', label='Random-walk Laplacian')
    plt.plot(k_values, obj_rw_r, color='red', linestyle='dashed', label='Random-walk Laplacian and normalized rows')

    plt.xlabel("Number of partitions")
    plt.ylabel("Objective function")
    plt.legend()
    plt.show()


# How to use
A = get_adjacency_matrix("../data/Oregon-1.txt")
labels, partitions = kmeans_partitions(k=10,
                                       normed_eigenvector_rows=False,
                                       laplacian_type="normalized",
                                       A=A,
                                       out_file="Oregon-1_partition.txt")

print(partition_sizes(labels))
print(objective(partitions, A))