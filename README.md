# blazarkit

**blazarkit** is a Python package for visualising and analysing pre-computed spectral fitting results from the Fermi/SDSS-V Blazar Analysis Pipeline.

> **Nlowie et al. (in prep.)** — *Optical Spectroscopic Analysis of Fermi Detected Blazars using SDSS-V*

---

## What is a blazar?

Blazars are a subclass of radio loud Active Galactic Nuclei (AGN) with relativistic jets of pointed directly toward Earth. They are among the most energetic objects in the Universe and are detected across the full electromagnetic spectrum from radio to gamma rays.

The SDSS automated spectroscopic pipeline systematically misclassifies blazars as stars because it has no model for the non-thermal jet emission. This pipeline corrects that by decomposing each optical spectrum into physically motivated components:

| Model | Physical meaning | Blazar class |
|-------|-----------------|--------------|
| Galaxy | Passive elliptical host galaxy | — |
| QSO | Accretion disc + broad emission lines | — |
| Powerlaw | Featureless non-thermal jet continuum | BL Lac |
| Powerlaw + Galaxy | Jet + host galaxy | BL Lac candidate |
| Powerlaw + QSO | Jet + accretion disc/BLR | FSRQ candidate |
| Powerlaw + Lines | Jet + emission lines only | FSRQ candidate |

**blazarkit** lets you load the pre-computed results and reproduce all diagnostic plots for any of the 707 sources in the sample.

---

## What you get

- **Spectral decomposition plots** — observed spectrum with best-fit model, jet/host component shading, line markers, and zoom insets on key spectral features
- **Redshift posterior plots** — chi-squared (z) curves and p(z) for all six model families
- **Equivalent width measurements** — for 16 emission and absorption features
- **Raw fit parameters** — redshift, jet fraction, power-law slope, AICc margin, and more
- **Classification summary** — one-line description of each source
- **Epoch comparison** — overlay spectra from multiple observations to detect variability
- **Population plots** — jet fraction vs redshift, PL slope distributions, and more
- **EW summary** — bar chart of all detected spectral lines with S/N labels

---

## Installation

### Requirements
- Python 3.8 or higher
- Works on Mac, Windows, and Linux

### Install blazarkit

```bash
pip install git+https://github.com/minlowie/Blazar-Analysis-Pipeline.git
```

### Install dependencies manually (if needed)

```bash
pip install numpy matplotlib astropy scipy
```

---

## Getting the data

### Cache

The pre-computed spectral fitting results cache will be publicly available on
Zenodo upon publication of SDSS-V DR20.

**Two datasets will be available:**

- **Main sample** (707 sources — primary science results)
- **Multiple-epoch sample** (966 observations — variability/changing-look analysis)

In the meantime, access is available upon request:

> Contact **Mohammed Nlowie Iddrisu** at m.i.nlowie@sms.ed.ac.uk

Once you have the cache, point `NAKBlaZarCache` to your local folder:

```python
cache = NAKBlaZarCache(cache_dir='/path/to/your/cache')
```

### The VAC table (optional)

The Value-Added Catalogue (VAC) is available through the SDSS Science Archive Server (SAS):

> **SDSS-V DR20 VAC: this is to be added upon publication**
> https://data.sdss.org/sas/dr20/ : 

The VAC contains the full sample of 707 blazar candidates with their classifications, redshifts, jet fractions, and continuum parameters. You can use it to select sources of interest before loading them from the cache.

---

## Quick start

```python
from blazarkit import NAKBlaZarCache, bz_inspect

# Point to your local cache folder
cache = NAKBlaZarCache(cache_dir='/path/to/your/cache')

# Load a source by SDSS_ID and MJD
results = cache.load('20570296', mjd=60027)

# Plot spectrum with best model
fig1, fig2 = bz_inspect(results, model='best')
```

---

## Sample output

![Example spectral fit](example.png)

**Figure description:** Multi-component optical spectral decomposition of a Fermi-detected BL Lac candidate. The Powerlaw+Galaxy model (red) separates the jet contribution (orange shading) from the host galaxy (green shading). The inset zooms in on the Ca II H&K absorption doublet used to anchor the redshift. The lower panel shows normalised residuals.

---

## Usage examples

```python
from blazarkit import NAKBlaZarCache, bz_inspect, normalize_shape

cache = NAKBlaZarCache(cache_dir='/path/to/SDSS-V_Fermi_Blazars_Cache')
results = cache.load('20570296', mjd=60027)

# Best model only — spectrum only (model is required)
fig1, fig2 = bz_inspect(results, model='best')

# All model fits overlaid
fig1, fig2 = bz_inspect(results, model='all')

# Specific models only
fig1, fig2 = bz_inspect(results, model=['best', 'Powerlaw+Galaxy', 'Powerlaw+QSO'])

# Also show chi2/p(z) figure
fig1, fig2 = bz_inspect(results, model='best', show_fig1=True)

# Minimal clean plot
fig1, fig2 = bz_inspect(results, model='best',
                              show_inset=False,
                              show_line_markers=False,
                              show_residuals=False)

# Zoom into a specific wavelength range
fig1, fig2 = bz_inspect(results, model='best', wavelength_range=(4000, 7000))

# Override y-axis limits
fig1, fig2 = bz_inspect(results, model='best', ylim=(-2, 30))

# Save to PDF
fig1, fig2 = bz_inspect(results, model='best', save_dir='my_plots')
```

---

## Function reference

### `bz_inspect`

```python
bz_inspect(results, model,
                save_dir=None,
                show_fig1=False,
                show_fig2=True,
                show_inset=True,
                show_residuals=True,
                show_components=True,
                show_line_markers=True,
                show_fraction_label=True,
                wavelength_range=(3600, 10400),
                ylim=None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `results` | — | Dict from `cache.load()` |
| `save_dir` | `None` | Directory to save PDF; if None, not saved |
| `model` | required | `'best'`, `'all'`, or list e.g. `['Powerlaw+Galaxy', 'Powerlaw+QSO']` |
| `show_fig1` | `False` | Show χ²(z) and p(z) diagnostic figure |
| `show_fig2` | `True` | Show best-fit spectrum figure |
| `show_inset` | `True` | Show zoom insets for detected lines only |
| `show_residuals` | `True` | Show normalised residuals panel |
| `show_components` | `True` | Show shaded host/jet component fills |
| `show_line_markers` | `True` | Show emission/absorption line markers |
| `show_fraction_label` | `True` | Show Host/Jet percentage text box |

| `wavelength_range` | `(3600, 10400)` | Wavelength range in Å |
| `ylim` | `None` | Override automatic y-axis limits as `(y_min, y_max)` |

---

### `NAKBlaZarCache`

```python
# Initialise with your local cache folder
cache = NAKBlaZarCache(cache_dir='/path/to/your/cache')

# Load a source
results = cache.load(sdss_id, mjd=mjd)

# Check if a source exists locally
cache.exists(sdss_id, mjd=mjd)

# List locally cached sources
cache.list_cached_objects()
```

---

### `measure_ew`

```python
results_ew = measure_ew(wave, flux, err, fit_mask, z)
```

Returns a list of tuples `(name, obs_wave, ew, ew_err, snr, detected, ltype, window)` for 16 spectral features. Detected if `S/N >= 3`.

---

### `normalize_shape`

```python
p_normalised = normalize_shape(p)
```

Normalises an array to its peak value for shape comparison plots.

---

### `get_redshift_peaks`

```python
peaks = get_redshift_peaks(pz_total, z_grid, min_height=0.05)
```

Returns a list of `(z, relative_height)` tuples for all peaks in the global p(z) posterior, sorted by height descending.


---

### `bz_classify`

```python
summary = bz_classify(results)
```

Prints and returns a one-line description of the classification of a source — blazar class, jet fraction, S/N, and key detected spectral lines.

---

### `bz_lines`

```python
fig = bz_lines(results, min_snr=3.0, save_dir=None)
```

Bar chart of all detected equivalent width measurements. Blue = emission, green = absorption. The |EW| = 5 Å BL Lac/FSRQ boundary is marked. S/N values are labelled on each bar.

---

### `bz_epochs`

```python
fig = bz_epochs(sdss_id, mjd_list, cache,
                     wavelength_range=(3600, 10400),
                     show_model=True,
                     save_dir=None)
```

Overlays spectra from multiple observations of the same source, coloured from dark (earliest) to bright (latest MJD). The lower panel shows the flux ratio relative to the first epoch. Useful for identifying changing-look blazar candidates.

| Parameter | Description |
|-----------|-------------|
| `sdss_id` | SDSS_ID of the source |
| `mjd_list` | List of MJD integers to compare |
| `cache` | NAKBlaZarCache instance |
| `wavelength_range` | Wavelength range in Å |
| `show_model` | Overlay best-fit model per epoch |
| `save_dir` | Save as PDF if provided |

---

### `bz_population`

```python
fig = bz_population(results_list,
                      x_param='z_best',
                      y_param='jet_frac',
                      color_by='best_label',
                      save_dir=None)
```

Population-level scatter plot and distribution from a list of cached results.

| Parameter | Options |
|-----------|---------|
| `x_param` | `'z_best'`, `'pl_alpha'`, `'pl_delta'`, `'aicc_margin'`, `'sn'`, `'jet_frac'` |
| `y_param` | Same as x_param |
| `color_by` | `'best_label'` (model family) or `'fermi_class'` |

**Example:**
```python
# Load all sources into a list first
results_list = []
for fname in sorted(cache.list_cached_objects()):
    parts   = fname.replace('.pkl.gz', '').split('_')
    results_list.append(cache.load(parts[1], mjd=int(parts[2])))

# Jet fraction vs redshift
fig = bz_population(results_list, x_param='z_best', y_param='jet_frac')

# PL alpha vs jet fraction coloured by Fermi class
fig = bz_population(results_list, x_param='pl_alpha', y_param='jet_frac',
                      color_by='fermi_class')
```

---

## Cache structure

Each `.pkl.gz` file contains a dictionary with the following structure:

```
results
├── metadata       — SDSS_ID, MJD, z_sdss, obj_class, fermi_class, sn_median
├── spectrum       — common_wave, flux_resamp, err_resamp, x_fit, y_fit, e_fit, fit_mask
├── chi2_grids     — z_grid, chi2 curves for all six model families
├── pz_results     — p(z) for each family, global p_total, z_map_total, C_global
├── lmfit_results  — best_label, z_best, z_err, pl_alpha, pl_delta, aicc_margin, ...
├── components     — jet fractions and component fluxes for combined models
├── ew_results     — equivalent width measurements for all 16 lines
└── config         — pipeline settings
```

---

## Spectral lines measured

**Emission:** Ly α (1215 Å), C IV (1549 Å), C III] (1909 Å), Fe II (2600 Å),
Mg II (2796 Å), [O II] (3729 Å), H β (4861 Å), [O III] (5007 Å),
H α (6563 Å), [N II] (6584 Å)

**Absorption:** Ca II K (3934 Å), Ca II H (3969 Å), Ca I G (4304 Å),
Mg b (5184 Å), Na I D (5893 Å), Ca Fe (5269 Å)

---

## Troubleshooting

**`ModuleNotFoundError`**
```bash
pip install numpy matplotlib astropy scipy
```

**`FileNotFoundError: No cache found`**
Check that your cache directory path is correct and the file exists in the format `obj_{SDSS_ID}_{MJD}.pkl.gz`. Run `cache.list_cached_objects()` to see available sources.

**Plots not showing in Jupyter**
Add `%matplotlib inline` at the top of your notebook.

**Zoom insets not appearing**
Insets only appear for lines detected at the required S/N. Ca II H&K requires S/N >= 5; Mg II and H-alpha require S/N >= 3. Run `measure_ew` to check detected lines.

**Windows path issues**
Use forward slashes or raw strings for paths:
```python
cache = NAKBlaZarCache(cache_dir=r'C:\Users\name\Downloads\SDSS-V_Fermi_Blazars_Cache')
```

---

## Citation

### Paper
If you use any results from this work, please cite:

```
Nlowie et al. (in prep.) — Optical Spectroscopic Analysis of Fermi Detected
Blazars using SDSS-V. MNRAS.
```

### Code
If you use blazarkit to visualise or analyse the pre-computed results, please also cite:

```
Nlowie et al. (2025) — blazarkit: Plotting and analysis utilities for
Fermi/SDSS-V blazar spectral fitting. Zenodo. doi:10.5281/zenodo.XXXXXXX
```

> Note: blazarkit is a visualisation and analysis tool for the pre-computed
> spectral fitting results. It does not include the fitting pipeline itself.
> To reproduce or extend the fitting, please contact the authors.

### Data
If you use the pre-computed spectral fitting results (cache files), please also cite:

**Main sample (707 sources):**
```
Nlowie et al. (2025) — Fermi/SDSS-V Blazar Spectral Fitting Results:
Main Sample. Zenodo. doi:10.5281/zenodo.YYYYYYY
```

**Multiple-match sample (966 observations, for variability analysis):**
```
Nlowie et al. (2025) — Fermi/SDSS-V Blazar Spectral Fitting Results:
Multiple-Epoch Sample. Zenodo. doi:10.5281/zenodo.ZZZZZZZ
```

> All DOIs will be confirmed and updated upon publication of SDSS-V DR20.
> In the meantime, contact m.i.nlowie@sms.ed.ac.uk for data access.

---

## Contact

| Name | Institution | Email |
|------|-------------|-------|
| Mohammed Nlowie Iddrisu (Lead) | University of Edinburgh / Royal Observatory of Edinburgh | m.i.nlowie@sms.ed.ac.uk |
| Prof. James Aird (Supervisor) | University of Edinburgh | james.aird@ed.ac.uk |
| Dr. Eli Kasai (Supervisor) | University of Namibia | ekasai@unam.na |

Funded by the **Development in Africa with Radio Astronomy (DARA)** programme.
