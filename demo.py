# %% [markdown]
# # Fermi/SDSS-V Blazar Analysis Pipeline — Demo
#
# This notebook demonstrates how to use the blazar spectral fitting pipeline
# to load pre-computed results from the cache and reproduce all diagnostic
# plots for any source in the sample.
#
# The pipeline fits six physically motivated spectral model families to
# BOSS optical spectra of Fermi-LAT blazar candidates, measuring optical
# jet fractions, spectroscopic redshifts, and equivalent widths.
#
# ### Before you start
# 1. Download the cache from Zenodo: <DOI — to be added upon publication>
# 2. Place Fits_with_Native_resampling_cache/ in the same directory
# 3. Install dependencies: pip install -r requirements.txt
#
# ### Reference
# Nlowie et al. (in prep.) — Optical Spectroscopic Analysis of Fermi
# Detected Blazars using SDSS-V

# %% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

sys.path.insert(0, '.')  # adjust if plot_from_cache.py is in a subfolder

from blazarkit import (
    RedshiftResultsCache,
    plot_from_cache,
    plot_multi_object_comparison_single,
    compute_EW_for_all_lines,
)

print('Imports successful.')

# %% [markdown]
# ## 1. Initialise the cache
#
# The cache contains one compressed file per source in the format
# obj_{SDSS_ID}_{MJD}.pkl.gz. Check the cache directory for available
# sources and their MJD values.

# %%
CACHE_DIR = 'Fits_with_Native_resampling_cache'

cache = RedshiftResultsCache(cache_dir=CACHE_DIR)
cached_files = cache.list_cached_objects()

# Show the first 10 files to identify available sources
print('\nFirst 10 cached sources:')
for f in sorted(cached_files)[:10]:
    print(f'  {f}')

# %% [markdown]
# ## 2. Load a single source
#
# Replace SDSS_ID and MJD with any values from the cache above.
# The SDSS_ID and MJD are encoded in the filename:
# obj_{SDSS_ID}_{MJD}.pkl.gz

# %%
SDSS_ID = 'your_sdss_id_here'   # e.g. '79336239'
MJD     = 0                      # e.g. 59955

results = cache.load_object_results(SDSS_ID, mjd=MJD)

# Print a summary of the source
meta  = results['metadata']
lmfit = results['lmfit_results']
comp  = results['components']

print(f"\n{'='*55}")
print(f"SDSS_ID:       {meta['SDSS_ID']}")
print(f"Fermi class:   {meta['fermi_class']}")
print(f"SDSS class:    {meta['obj_class']}")
print(f"Best model:    {lmfit['best_label']}")
print(f"z_fit:         {lmfit['z_best']:.4f}")
print(f"z_SDSS:        {meta['z_sdss']}")
print(f"AICc margin:   {lmfit['aicc_margin']:.1f}")
print(f"PL alpha:      {lmfit['pl_alpha']:.4f}")
print(f"PL delta:      {lmfit['pl_delta']:.4f}")
print(f"S/N:           {meta['sn_median']:.1f}")

# Jet fraction (for combined models)
best = lmfit['best_label']
if best == 'Powerlaw+Galaxy' and comp['galpl_contrib'] is not None:
    print(f"Jet fraction:  {comp['galpl_contrib']['frac_pl']*100:.1f}%")
    print(f"Host fraction: {comp['galpl_contrib']['frac_tpl']*100:.1f}%")
elif best.startswith('Powerlaw+QSO') and comp['qsopl_contrib'] is not None:
    print(f"Jet fraction:  {comp['qsopl_contrib']['frac_pl']*100:.1f}%")
    print(f"Disc/BLR:      {comp['qsopl_contrib']['frac_tpl']*100:.1f}%")
elif best.startswith('Powerlaw+Lines') and comp['linepl_contrib'] is not None:
    print(f"Jet fraction:  {comp['linepl_contrib']['frac_pl']*100:.1f}%")
print(f"{'='*55}")

# %% [markdown]
# ## 3. Default plot — all panels
#
# By default this produces two figures:
# - Figure 1: chi2(z) curves and redshift posterior p(z) for all six model families
# - Figure 2: best-fit spectrum with component decomposition, line markers,
#             zoom insets for detected lines, and normalised residuals

# %%
fig1, fig2 = plot_from_cache(results)

# %% [markdown]
# ## 4. Customising the plot
#
# All panels are individually controllable via keyword arguments.

# %% Only the spectrum — skip the chi2/p(z) figure
fig1, fig2 = plot_from_cache(results, show_fig1=False)

# %% Minimal clean plot — best model only, no extras
fig1, fig2 = plot_from_cache(
    results,
    show_fig1        = False,
    show_inset       = False,
    show_line_markers= False,
    show_residuals   = False,
    show_fraction_label = False,
)

# %% Show all model fits overlaid on the spectrum
fig1, fig2 = plot_from_cache(results, models_to_show='all')

# %% Show specific models only
fig1, fig2 = plot_from_cache(
    results,
    show_fig1      = False,
    models_to_show = ['Powerlaw+Galaxy', 'Galaxy', 'Powerlaw'],
)

# %% Zoom into a specific wavelength range
fig1, fig2 = plot_from_cache(
    results,
    show_fig1        = False,
    wavelength_range = (4000, 7000),
)

# %% Override y-axis limits
fig1, fig2 = plot_from_cache(
    results,
    show_fig1 = False,
    ylim      = (-2, 30),
)

# %% Save the spectral plot to a PDF
fig1, fig2 = plot_from_cache(results, show_fig1=False, save_dir='output_plots')

# %% [markdown]
# ## 5. Equivalent width measurements
#
# Equivalent widths are measured for 16 emission and absorption features
# using an adaptive integration window that maximises S/N.
# A line is considered detected if S/N >= 3.

# %%
spec   = results['spectrum']
z_best = results['lmfit_results']['z_best']

results_ew = compute_EW_for_all_lines(
    spec['common_wave'],
    spec['flux_resamp'],
    spec['err_resamp'],
    spec['fit_mask'],
    z_best
)

print(f"Equivalent widths at z = {z_best:.4f}:")
print(f"{'Line':<15} {'Type':<12} {'EW (A)':>10} {'±err':>8} {'S/N':>8} {'Det?':>6} {'Window':>8}")
print("-" * 70)
for name, obs, ew, ew_err, snr, detected, ltype, window in results_ew:
    if np.isfinite(ew):
        flag   = 'yes' if detected else ''
        win_str = f'{window} A' if window is not None else ''
        print(f"{name:<15} {ltype:<12} {ew:>10.2f} {ew_err:>8.2f} "
              f"{snr:>8.1f} {flag:>6} {win_str:>8}")

# %% [markdown]
# ## 6. Access raw fit parameters
#
# All best-fit parameters from lmfit are stored in the cache and can be
# accessed directly for custom analysis.

# %%
lmfit = results['lmfit_results']

print("Best-fit parameters:")
print(f"{'Name':<15} {'Value':>12} {'Stderr':>12}")
print("-" * 42)
for name, param in lmfit['best_fit_params']['params'].items():
    val     = param['value']
    err     = param['stderr']
    err_str = f"{err:.4f}" if err is not None else 'N/A'
    print(f"{name:<15} {val:>12.4f} {err_str:>12}")

print(f"\nchi2:        {lmfit['best_fit_params']['chisqr']:.2f}")
print(f"redchi:      {lmfit['best_fit_params']['redchi']:.3f}")
print(f"AICc margin: {lmfit['aicc_margin']:.1f}")
print(f"PL alpha:    {lmfit['pl_alpha']:.4f}")
print(f"PL delta:    {lmfit['pl_delta']:.4f}")

# %% [markdown]
# ## 7. Accessing the redshift posterior
#
# The full p(z) posterior is stored for each source and can be used for
# custom redshift analysis or visualisation.

# %%
pz     = results['pz_results']
z_grid = results['chi2_grids']['z_grid']

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(z_grid, pz['p_total'],   'k-',  lw=2.5, label='Global p(z)')
ax.plot(z_grid, pz['pz_galpl'],  'r-',  lw=1.5, alpha=0.7, label='PL+Galaxy')
ax.plot(z_grid, pz['pz_qsopl'],  'm-',  lw=1.5, alpha=0.7, label='PL+QSO')
ax.plot(z_grid, pz['pz_linepl'], 'c-',  lw=1.5, alpha=0.7, label='PL+Lines')
ax.plot(z_grid, pz['pz_gal'],    'g-',  lw=1.2, alpha=0.5, label='Galaxy')
ax.plot(z_grid, pz['pz_qso'],    'b-',  lw=1.2, alpha=0.5, label='QSO')
ax.plot(z_grid, pz['pz_pl'],     color='orange', lw=1.2, alpha=0.5, label='PL')
ax.axvline(pz['z_map_total'], color='red', ls=':', lw=2,
           label=f"z_MAP = {pz['z_map_total']:.3f}")
ax.set_xscale('log')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.set_title(f"Redshift posterior — SDSS_ID {results['metadata']['SDSS_ID']}")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Loop over all sources in the cache
#
# You can iterate over all cached sources to extract statistics or
# reproduce plots in bulk.

# %%
# Extract key parameters from all cached sources
summary = []

for fname in sorted(cache.list_cached_objects()):
    # Parse SDSS_ID and MJD from filename
    parts   = fname.replace('.pkl.gz', '').split('_')
    sdss_id = parts[1]
    mjd     = int(parts[2]) if len(parts) > 2 else None

    try:
        res   = cache.load_object_results(sdss_id, mjd=mjd)
        lm    = res['lmfit_results']
        m     = res['metadata']
        c     = res['components']

        jet_frac = np.nan
        if lm['best_label'] == 'Powerlaw+Galaxy' and c['galpl_contrib']:
            jet_frac = c['galpl_contrib']['frac_pl']
        elif lm['best_label'].startswith('Powerlaw+QSO') and c['qsopl_contrib']:
            jet_frac = c['qsopl_contrib']['frac_pl']
        elif lm['best_label'].startswith('Powerlaw+Lines') and c['linepl_contrib']:
            jet_frac = c['linepl_contrib']['frac_pl']

        summary.append({
            'SDSS_ID'    : sdss_id,
            'MJD'        : mjd,
            'best_label' : lm['best_label'],
            'z_best'     : lm['z_best'],
            'z_sdss'     : m['z_sdss'],
            'fermi_class': m['fermi_class'],
            'jet_frac'   : jet_frac,
            'pl_alpha'   : lm['pl_alpha'],
            'aicc_margin': lm['aicc_margin'],
            'sn'         : m['sn_median'],
        })
    except Exception as e:
        print(f"  Skipped {fname}: {e}")

print(f"\nLoaded {len(summary)} sources")

# Convert to arrays for analysis
z_best   = np.array([s['z_best']   for s in summary])
jet_frac = np.array([s['jet_frac'] for s in summary])
pl_alpha = np.array([s['pl_alpha'] for s in summary])

print(f"Median z_best:    {np.nanmedian(z_best):.3f}")
print(f"Median jet frac:  {np.nanmedian(jet_frac[np.isfinite(jet_frac)]):.3f}")
print(f"Median PL alpha:  {np.nanmedian(pl_alpha[np.isfinite(pl_alpha)]):.3f}")

# %% [markdown]
# ## 9. Multi-object comparison plot
#
# Plot multiple sources side by side for visual comparison.
# Requires the main blazar sample table (VAC).

# %%
from astropy.table import Table
from astropy.io import fits

# Load the VAC table — update path as needed
vac = Table(fits.open('fermi_blazar_vac_dr20.fits')[1].data)
print(f"VAC: {len(vac)} sources")

# Select a few sources to compare — e.g. high S/N BL Lac candidates
# Adjust column names to match your VAC
object_list = [
    {'SDSS_ID': int(row['SDSS_ID']), 'MJD': None}
    for row in vac[:4]   # replace with your selection
]

fig = plot_multi_object_comparison_single(
    object_list = object_list,
    cache       = cache,
    final_bl    = vac,
    n_cols      = 2,
    save_dir    = 'output_plots'
)
