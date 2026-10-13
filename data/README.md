# Data

The images come from the **Alzheimer's Disease Neuroimaging Initiative
(ADNI)** and are **not included in this repository**. The ADNI Data Use
Agreement does not allow redistribution. To reproduce the analysis:

1. Request access at <https://adni.loni.usc.edu> ("Access Data and Samples").
2. Download tau-PET scans (e.g. flortaucipir / AV-1451) with their diagnostic
   labels (CN, MCI, AD).
3. Register them to a common template, extract one 2D slice per view, and save
   the slices as 8-bit grayscale PNGs with this layout:

```
data/tau_data/
├── tau_data_axial/
│   ├── AD/   <subject_id>_slice<k>.png
│   ├── CN/
│   └── MCI/
├── tau_data_coronal/   (same structure)
└── tau_data_sagittal/  (same structure)
```

The course provided this dataset already preprocessed: 1,093 subjects (631
CN, 356 MCI, 106 AD), one slice per view, at 182×218 pixels (axial,
sagittal) and 182×182 pixels (coronal).

`data/processed/` is created by the notebooks to cache extracted radiomics
features. Everything under `data/` except this file is git-ignored.
