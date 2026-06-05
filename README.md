# Blazar Analysis Pipeline

Plotting and analysis utilities for the multi-component spectral fitting pipeline developed in:

> **Nlowie et al. (in prep.)** — *Optical Spectroscopic Analysis of Fermi Detected Blazars using SDSS-V*

This repository contains the code needed to load pre-computed fitting results from the cache and reproduce all diagnostic plots for any source in the sample.

---

## What this pipeline does

The pipeline fits six physically motivated spectral model families to optical BOSS spectra of Fermi-LAT blazar candidates cross-matched with SDSS-V DR20:

- **Galaxy** — passive elliptical host galaxy (SWIRE templates)
- **QSO** — accretion disc + broad emission lines (QSOGEN templates)
- **Powerlaw** — featureless non-thermal jet continuum
- **Powerlaw + Galaxy** — jet + host galaxy → **BL Lac candidates**
- **Powerlaw + QSO** — jet + accretion disc/BLR → **FSRQ candidates**
- **Powerlaw + Lines** — jet + emission lines only → **FSRQ candidates**

Model selection uses the corrected Akaike Information Criterion (AICc). Redshifts are estimated from a Bayesian posterior constructed over a logarithmic grid z = 0.01–5.0.

---

## Repository structure

```
Blazar-Analysis-Pipeline/
├── plot_from_cache.py   ← all plotting and EW measurement functions
├── demo.py              ← demo script showing how to use the cache
├── requirements.txt     ← Python dependencies
└── README.md
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the cache

The pre-computed fitting results (746 sources) are available on Zenodo:

> **DOI: to be added upon publication**

Download and place the `Fits_with_Native_resampling_cache/` folder in the same directory as the scripts.

### 3. Run the demo

```bash
python demo.py
```

Or open `demo.py` in Jupyter (requires jupytext):

```bash
pip install jupytext
jupytext --to notebook demo.py
jupyter notebook demo.ipynb
```

---

## Usage

```python
from plot_from_cache import RedshiftResultsCache, plot_from_cache

# Load cache
cache = RedshiftResultsCache(cache_dir='Fits_with_Native_resampling_cache')

# Load a source by SDSS_ID and MJD
results = cache.load_object_results('79336239', mjd=59955)

# Plot all diagnostic panels
fig1, fig2 = plot_from_cache(results, save_dir=None)
```

---

## Sample output

The pipeline produces two diagnostic figures per source:

**Figure 1** — χ²(z) curves and redshift posterior p(z) for all six model families

**Figure 2** — Best-fit spectrum with component decomposition (jet contribution in orange, host/disc contribution in green/blue), spectral line markers, zoom insets on key features (Ca II H&K, [O II], H-alpha), and normalised residuals

---

## Data

- **Spectra**: SDSS-V DR20 BOSS spectroscopy (not yet public — available upon DR20 public release)
- **Cache**: Pre-computed fitting results available on Zenodo (DOI above)
- **VAC table**: `fermi_blazar_vac_dr20.fits` — available on Zenodo or something else -- still thinking ):

---

## Citation

If you use this code or the associated catalogue, please cite:

```
Nlowie et al. (in prep.) — Optical Spectroscopic Analysis of Fermi Detected Blazars using SDSS-V
```

---

## Contact

Mohammed Nlowie Iddrisu  
University of Edinburgh / Royal Observatory of Edinburgh  
Funded by the Development in Africa with Radio Astronomy (DARA) programme
