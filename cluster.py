"""
Various methods to cluster embeddings.
"""

from distinctipy import distinctipy
import matplotlib as mpl
import numpy as np
import os
import scipy.linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import spectral as spy


class Cluster:
    """ Represents a clustering layer of single hyperspectral image.

    Parameters
    ----------
    cube : HyperCube
        HyperCube with `emb` to cluster.

    Attributes
    ----------
    cube : HyperCube
        See above.
    clus : (A,B) ndarray
        -1 for unclustered areas and a consecutive natural number for each other clustering.
    """

    def __init__(self, cube):
        self.cube = cube
        self.clus = np.full(cube.cube.shape[:2], -1, dtype=np.int16)

    def k_means(self, n_clusters):
        """ Performs k-means clustering on embedding. 

        Parameters
        ----------
        n_clusters : int
            Number of clusters to produce.

        """

        self.clus[self.cube.mask] = KMeans(n_clusters=n_clusters).fit(
            self.cube.emb[self.cube.mask]).labels_

    def gaussian_mixture(self, n_clusters):
        """ Performs gaussian mixture model clustering on embedding. 

        Parameters
        ----------
        n_clusters : int
            Number of clusters to produce.

        """

        self.clus[self.cube.mask] = GaussianMixture(
            n_components=n_clusters).fit_predict(self.cube.emb[self.cube.mask])

    # TODO: Make safe
    def hierarchical_gaussian_mixture(self, depth):
        """ Performs hierarchical gaussian mixture model clustering on
        embedding by using a series of two-cluster gaussian mixture models.

        Parameters
        ----------
        depth : int
            2**depth clusters will be produced.

        """

        def _hierarchical_gaussian_mixture(emb, depth):
            clus = GaussianMixture(
                n_components=2, random_state=0).fit_predict(emb)
            if depth:
                i_l = clus == 0
                i_r = clus == 1
                clus[i_l] = _hierarchical_gaussian_mixture(emb[i_l], depth-1)
                clus[i_r] = _hierarchical_gaussian_mixture(
                    emb[i_r], depth-1)+np.max(clus)+1

            return clus

        self.clus[self.cube.mask] = _hierarchical_gaussian_mixture(
            self.cube.emb[self.cube.mask], depth)

    def combine_spectrally_similar(self, n_clusters):
        """ Iteratively combines clusters with the smallest spectral angle
        between their mean pixels.

        Parameters
        ----------
        n_clusters : int
            Stopping condition for when to stop combining clusters.

        """

        angs = Cluster.get_angs(self.clus, self.cube.cube)
        X = self.cube.cube[self.cube.mask]
        clus = self.clus[self.cube.mask]

        while len(np.unique(clus)) > n_clusters:
            new, old = np.unravel_index(np.argmin(angs), angs.shape)

            clus[clus == old] = new
            angs[old] = np.pi
            angs[:, old] = np.pi

            u = np.mean(X[clus == new], axis=0)
            for i in range(len(angs)):
                v = np.mean(X[clus == i], axis=0)
                ang = Cluster.get_ang(u, v)

                if angs[new, i] < np.pi:
                    angs[new, i] = ang
                elif angs[i, new] < np.pi:
                    angs[i, new] = ang
        self.clus[self.cube.mask] = clus

    def save_clustering(self, file_path, color=False):
        """ Save clustering image to `file_path`.

        Parameters
        ----------
        file_path : PathLike
            Output path for `clus`.
        color : bool
            Whether to apply a false coloring to the clustering.
            WISER is not currently able to do so, so this is helpful.

        """

        clus = self.clus
        if color:
            colors = ['#000000'] + distinctipy.get_colors(1+np.max(self.clus))
            clus = mpl.colors.ListedColormap(colors)(self.clus+1)

        if os.path.splitext(file_path)[1] == '.npy':
            np.save(file_path, clus)
        elif os.path.splitext(file_path)[1] == '.npz':
            np.savez(file_path, clus)
        else:
            spy.envi.save_classification(file_path, clus, force=True)

    @staticmethod
    def get_ang(u, v):
        """ Computes the spectral angle between two vectors. 

        Parameters
        ----------
        u : (A,) ndarray
            Mean pixel of first cluster.
        v : (A,) ndarray
            Mean pixel of second cluster.

        Returns
        -------
        theta : float
            Angle between `u` and `v`.

        """

        return np.arccos(np.dot(u, v)/(1e-8+scipy.linalg.norm(u)*scipy.linalg.norm(v)))

    # Requires clus with classes 0...n
    @staticmethod
    def get_angs(clus, X):
        """ Computes lower triangular matrix of spectral angle between every
        pairing of clusters. 

        Parameters
        ----------
        clus : (A,) ndarray
            Clustering map of classes 0...n.
        X : (A, B) ndarray
            Corresponding spectral data for clustering map.

        Returns
        -------
        angs : (MAX(clus)+1, MAX(clus)+1) ndarray
            Lower triangular matrix holding spectral angle between every
            pairing of clusters.

        """

        n_clusters = np.max(clus)+1
        means = np.array([np.mean(X[clus == i], axis=0)
                         for i in np.arange(n_clusters)])

        angs = np.full((n_clusters, n_clusters), np.pi, dtype=np.float64)

        for i in range(n_clusters):
            u = means[i]
            for j in range(i):
                v = means[j]
                angs[j, i] = Cluster.get_ang(u, v)

        return angs
