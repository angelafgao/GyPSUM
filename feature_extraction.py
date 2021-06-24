"""
Various methods to process hyperspectral images to prepare for clustering analysis
"""

import numpy as np
import os
from pysptools.material_count import HySime
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import spectral as spy
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model


class HyperCube:
    """ Represents a single hyperspectral image and its transformation.

    Parameters
    ----------
    img_path : PathLike
        Path to hyperspectral ENVI image.
    mask_path : PathLike, optional
        Path to mask in either npy or ENVI image format.

    Attributes
    ----------
    cube : (A,B,C) ndarray
        Stores spectral information.
    bands : (C,) ndarray
        Stores wavelength labels for `cube`.
    mask : (A,B) ndarray
        Masks valid pixels in `cube`.
    n_components : int
        Dimension of embedding space to produce.
    emb : (A,B,n_components) ndarray
        Generated latent representation of cube.

    """

    def __init__(self, img_path, mask_path=None):
        fp = spy.open_image(img_path)

        self.cube = np.array(fp.load())
        self.bands = np.array(fp.bands.centers)

        if mask_path is not None:
            self.mask = np.squeeze(self.load_file(mask_path)).astype(np.bool)
        else:
            self.mask = np.ones(self.cube.shape[:-1], dtype=np.bool)

        self.n_components = None
        self.emb = None

    def spectral_subset(self, band_min=-np.inf, band_max=np.inf):
        """ Trims `cube` and `bands` to spectral range. 

        Parameters
        ----------
        band_min : float, optional
            Lower bound on acceptable band wavelength.
        band_max : float, optional
            Upper bound on acceptable band wavelength.

        """

        valid_bands = (self.bands >= band_min) & (self.bands <= band_max)

        self.cube = self.cube[..., valid_bands]
        self.bands = self.bands[valid_bands]

    def unmask_value(self, value):
        """ Removes pixels in `mask` where all values in `cube` equal `value`.

        Parameters
        ----------
        value : float
            Value to unmask.

        """

        self.mask &= np.logical_not(
            np.all(np.isclose(self.cube, value), axis=-1))

    def clip(self, min=0, max=1):
        """ Clips `cube` values to range [`min`, `max`].

        Parameters
        ----------
        min : float, optional
            Minimum spectral value.
        max : float, optional
            Maximum spectral value.

        """

        np.clip(self.cube, min, max, out=self.cube)

    def ratio(self, ratio_path):
        """ Divide `cube` by ratio.

        Parameters
        ----------
        ratio_path : PathLike
            Path to ratio file in either npy or ENVI image format.
            Loaded ratio has dimensions (C,).

        """

        self.cube /= self.load_file(ratio_path)

    def normalize(self):
        """ Divide `cube` by per-pixel l2-norm.
        """

        self.cube /= scipy.linalg.norm(self.cube, axis=-1)[..., np.newaxis]

    # spy.remove_continuum outputs float64 so `out` is not used
    def remove_continuum(self):
        """ Remove continuum from `cube`.
        For additional info, see spectral.remove_continuum at:
        https://github.com/spectralpython/spectral/blob/master/spectral/algorithms/continuum.py

        """

        self.cube = spy.remove_continuum(self.cube, self.bands)
        self.cube[np.isnan(self.cube)] = 0

    def standardize(self):
        """ Mean-subtract each pixel in `cube` then scale to unit variance.
        For additional info, see sklearn.preprocessing.StandardScaler at:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        """

        self.cube[self.mask] = StandardScaler(
        ).fit_transform(self.cube[self.mask])

    def hysime(self):
        """ Compute HySime signal subspace estimate and store as `n_components`.
        For additional info, see pysptools.material_count.hysime at:
        https://pysptools.sourceforge.io/material_count.html#hyperspectral-signal-subspace-identification-by-minimum-error-hysime

        """

        self.n_components = HySime().count(self.cube)[0]

    def set_n_components(self, n_components):
        """ See `n_components` in HyperCube#Attributes.
        """

        self.n_components = n_components

    def pca(self):
        """ Transform `cube` into its principal-component representation of
        size `n_components` and store in `emb`.

        """

        self.emb = np.full(
            self.cube.shape[:-1] + (self.n_components, ), -1, dtype=np.float32)
        self.emb[self.mask] = PCA(n_components=self.n_components).fit_transform(
            self.cube[self.mask, ...])

    def autoencoder(self, epochs=10):
        """ Transform `cube` into a learned autoencoder latent space.
        For more info, refer to `Autoencoder`.

        """

        X = self.cube[self.mask]
        autoencoder = Autoencoder(self.n_components, len(self.bands))
        autoencoder.compile(optimizer='adam', loss=SpectralAngleLoss())
        autoencoder.fit(X, X,
                        epochs=epochs,
                        shuffle=True)

        self.emb = np.full(
            self.cube.shape[:-1] + (self.n_components, ), -1, dtype=np.float32)
        self.emb[self.mask] = autoencoder.transform(X)

    def save_emb(self, file_path):
        """ Save `emb` image to `file_path`.

        Parameters
        ----------
        file_path : PathLike
            Output path for `clus`.

        """

        if os.path.splitext(file_path)[1] == '.npy':
            np.save(file_path, self.emb)
        elif os.path.splitext(file_path)[1] == '.npz':
            np.savez(file_path, self.emb)
        else:
            spy.envi.save_image(file_path, self.emb, force=True)

    def save_cube(self, file_path):
        """ Save `cube` image to `file_path`.

        Parameters
        ----------
        file_path : PathLike
            Output path for `clus`.

        """

        if os.path.splitext(file_path)[1] == '.npy':
            np.save(file_path, self.cube)
        elif os.path.splitext(file_path)[1] == '.npz':
            np.savez(file_path, self.cube)
        else:
            spy.envi.save_image(file_path, self.cube, force=True)

    def save_mask(self, file_path):
        """ Save `mask` to `file_path`.

        Parameters
        ----------
        file_path : PathLike
            Output path for `clus`.

        """

        if os.path.splitext(file_path)[1] == '.npy':
            np.save(file_path, self.mask)
        elif os.path.splitext(file_path)[1] == '.npz':
            np.savez(file_path, self.mask)
        else:
            spy.envi.save_classification(
                file_path, self.mask.astype(np.uint8), force=True)

    @staticmethod
    def load_file(file_path):
        """ Load and return file either from npy or ENVI image format.

        Parameters
        ----------
        file_path : PathLike
            Path to file to load.

        Returns
        -------
        data : ndarray
            Loaded data.

        """

        if os.path.splitext(file_path)[1] in ['.npy', '.npz']:
            return np.load(file_path)
        else:
            return spy.open_image(file_path).load()


class Autoencoder(Model):
    """ Deep model to decrease data dimensionality.

    Parameters
    ----------
    n_components : int
        Size of latent representation.
    n_bands : int
        Input/output spectral dimensionality.

    Attributes
    ----------
    encoder : tf.keras.Sequential
        Network to encode spectral data to latent embedding.
    decoder : tf.keras.Sequential
        Network to decode spectral data from latent embedding.

    """

    def __init__(self, n_components, n_bands):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(n_components),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(24, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_bands),
        ])

    def call(self, X):
        """ Encode then decode input `X`.

        Parameters
        ----------
        X : (A,n_bands) ndarray
            Vector of spectral pixels to be transformed.

        Returns
        -------
        Xhat : (A,n_bands) ndarray
            Encoded then decoded `X`.

        """

        return self.decoder(self.encoder(X))

    def transform(self, X):
        """ Encode input `X`.

        Parameters
        ----------
        X : (A,n_bands) ndarray
            Vector of spectral pixels to be transformed.

        Returns
        -------
        Z : (A,n_components) ndarray
            Encoded `X`.

        """

        return self.encoder(X)


class SpectralAngleLoss(Loss):
    """ Autoencoder loss function implementing the spectral-angle-mapping
    algorithm.

    """

    def call(self, y_true, y_pred):
        y_true = tf.math.l2_normalize(y_true, axis=-1)
        y_pred = tf.math.l2_normalize(y_pred, axis=-1)
        return tf.math.acos(tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1))
