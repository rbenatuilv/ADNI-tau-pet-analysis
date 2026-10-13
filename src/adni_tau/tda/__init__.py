"""Topological data analysis (TDA) pipeline for AD vs CN classification.

Every 2D slice is turned into persistence diagrams (H0 = connected
components, H1 = loops) through one of several filtrations, the diagrams are
vectorised, and the vectors are classified.

Sub-modules
-----------
config          tunable constants (thresholds, radii, resolutions, split sizes)
data            loading of the balanced AD/CN train/test split, per view
preprocessing   thresholding, point clouds, adaptive (quantile) thresholds
topology        filtrations, persistence diagrams and their vectorisations
                (persistence landscape, Betti curve, lifetime curve)
templates       class-average templates and nearest-template distances
classification  nearest-template rule, multi-view logistic regression,
                coefficient analysis, cross-validation
pca_analysis    PCA on the topological feature matrix (+ Wald p-values)
filtration_3d   filtrations on a 3D (row, column, intensity) point cloud
plotting        visualisation helpers

Filtrations
-----------
``lower_star``     sublevel-set filtration of the pixel grid (cubical complex)
``rips``           Vietoris-Rips complex on above-threshold pixels
``delaunaycech``   Delaunay-Cech complex (same persistence as Cech, much faster)
``alpha``          alpha complex
``weighted_rips``  Rips complex where each point is weighted by its intensity
"""
