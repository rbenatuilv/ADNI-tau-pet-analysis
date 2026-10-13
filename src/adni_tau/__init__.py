"""Statistical and topological analysis of tau-PET brain slices (ADNI).

Sub-modules
-----------
paths               project directories and figure/table saving helpers
data                ``ImageLoader``: indexes and loads the PNG slices
stats               per-component group tests (ANOVA / Welch ANOVA)
roi                 brain mask and PCA/ANOVA-driven region of interest
radiomics_features  2D radiomics feature extraction (pyradiomics)
evaluation          grid-search + held-out evaluation helpers
tda                 topological data analysis pipeline (persistent homology)
"""

__version__ = "1.0.0"
