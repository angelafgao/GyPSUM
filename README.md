# Generalized Unsupervised Clustering of Hyperspectral Images of Geological Targets in the Near Infrared (GyPSUM)

<!-- ## Examples -->

<!-- **Arguments:** -->

<!--     General:
    * lr (float) - learning rate
    * clip (float) - threshold for gradient clip
    * n_epoch (int) - number of epochs
    * npix (int) - size of reconstruction images (npix * npix)
    * n_flow (int) - number of affine coupling blocks
    * logdet (float) - weight of the entropy loss (larger means more diverse samples)
    * save_path (str) - folder that saves the learned DPI normalizing flow model

    For radio interferometric imaging:
    * obspath (str) - observation data file

    For compressed sensing MRI:
    * impath (str) - fast MRI image for generating MRI measurements
    * maskpath (str) - compressed sensing sampling mask
    * sigma (float) - additive measurement noise
   -->
## Requirements
General requirements for clustering models:
* [pytorch](https://pytorch.org/)
* [sklearn](https://scikit-learn.org/)

For loading hyperspectral image data:
* [spectral](https://pypi.org/project/spectral/)

Please check ```environment.yml``` for the full Anaconda environment information.

## Citation
Angela F. Gao, Brandon Rasmussen, Peter Kulits, Eva L. Scheller, Rebecca Greenberger, Bethany L. Ehlmann; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2021, pp. 4294-4303
