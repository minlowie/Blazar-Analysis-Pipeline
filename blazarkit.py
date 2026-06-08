"""
blazarkit.py
============
Plotting and analysis utilities for the Fermi/SDSS-V Blazar Analysis Pipeline.

Loads pre-computed spectral fitting results from the cache and reproduces
all diagnostic plots without re-running the fitting pipeline.

Usage
-----
    from blazarkit import RedshiftResultsCache, plot_from_cache
    from blazarkit import plot_multi_object_comparison_single

    cache   = RedshiftResultsCache(cache_dir="Fits_with_Native_resampling_cache")
    results = cache.load_object_results("79336239", mjd=59955)
    plot_from_cache(results)

Requirements
------------
    numpy, matplotlib, astropy, scipy
    pip install numpy matplotlib astropy scipy

Reference
---------
    Nlowie et al. (in prep.)
    Cache data available at: <Zenodo DOI - to be added upon publication>
"""

import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from pathlib import Path


# -- Line lists (rest-frame wavelengths in Angstrong) ---------------------------

EMISSION_LINES = {
    'Ly_alpha': 1215,
    'C IV':     1549,
    'C III':    1909,
    'Fe II':    2600,
    'Mg II':    2796,
    '[O II]':   3729,
    'H_beta':   4861.3,
    '[O III]':  5007,
    'H_alpha':  6562.80,
    '[N II]':   6583.6,
}

ABSORPTION_LINES = {
    'Ca II K': 3933.7,
    'Ca II H': 3968.5,
    'Ca I G':  4304.40,
    'Mg_b':    5184,
    'Na I D':  5892.5,
    'Ca Fe':   5269,
}


# -- Cache system --------------------------------------------------------------
class RedshiftResultsCache:
    """
    Load and save spectral fitting results cached as compressed pickle files.

    Each source is stored as a single .pkl.gz file named
    obj_{SDSS_ID}_{MJD}.pkl.gz in the cache directory.
    """

    def __init__(self, cache_dir="Fits_with_Native_resampling_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_object_path(self, sdss_id, mjd=None):
        filename = f"obj_{sdss_id}_{mjd}.pkl.gz" if mjd is not None \
                   else f"obj_{sdss_id}.pkl.gz"
        return os.path.join(self.cache_dir, filename)

    def save_object_results(self, sdss_id, results_dict, mjd=None):
        results_dict['metadata']['timestamp'] = datetime.now().isoformat()
        filepath = self._get_object_path(sdss_id, mjd)
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {os.path.basename(filepath)}")
        return filepath

    def load_object_results(self, sdss_id, mjd=None):
        filepath = self._get_object_path(sdss_id, mjd)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No cache found: {filepath}")
        try:
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
        except gzip.BadGzipFile:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
        print(f"Loaded: {os.path.basename(filepath)}")
        return results

    def exists(self, sdss_id, mjd=None):
        return os.path.exists(self._get_object_path(sdss_id, mjd))

    def list_cached_objects(self):
        files = [f for f in os.listdir(self.cache_dir)
                 if f.startswith("obj_") and f.endswith(".pkl.gz")]
        print(f"Found {len(files)} cached objects in {self.cache_dir}/")
        return files


# --- Line marking -------------------------------------------------------------

def mark_spectral_lines(ax, z, obs_range=(3600, 10400)):
    """Mark emission and absorption lines at redshift z on axes ax."""
    ylim    = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_em    = ylim[0] + 0.95 * y_range
    y_ab    = ylim[0] + 0.10 * y_range

    for name, rest_wave in EMISSION_LINES.items():
        obs_wave = rest_wave * (1 + z)
        if obs_range[0] <= obs_wave <= obs_range[1]:
            ax.axvline(obs_wave, color='dodgerblue', ls='-', lw=1.6,
                       alpha=0.9, zorder=2)
            ax.text(obs_wave, y_em, f'{name}\n↓', rotation=90, fontsize=7,
                    va='bottom', ha='center', color='dodgerblue',
                    alpha=0.75, weight='bold', zorder=3)

    for name, rest_wave in ABSORPTION_LINES.items():
        obs_wave = rest_wave * (1 + z)
        if obs_range[0] <= obs_wave <= obs_range[1]:
            ax.axvline(obs_wave, color='forestgreen', ls='-', lw=1.6,
                       alpha=0.9, zorder=2)
            ax.text(obs_wave, y_ab, f'↑\n{name}', rotation=90, fontsize=7,
                    va='top', ha='center', color='forestgreen',
                    alpha=0.75, weight='bold', zorder=3)


def add_line_markers_to_plot(ax, z_fit, x_range=None):
    """Add emission and absorption line markers and update legend."""
    if x_range is None:
        x_range = ax.get_xlim()
    mark_spectral_lines(ax, z_fit, obs_range=x_range)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color='dodgerblue',  ls='--', lw=1.2, alpha=0.6),
        Line2D([0], [0], color='forestgreen', ls='--', lw=1.2, alpha=0.6),
    ])
    labels.extend(['Emission lines', 'Absorption lines'])
    ax.legend(handles, labels, fontsize=9, loc='upper right')


# -- EW measurement ------------------------------------------------------------

def measure_equivalent_width_hybrid_normalized(
        wave, flux, err, fit_mask, line_center,
        line_type='emission', min_snr=3.0, window_options=None):
    """
    Measure equivalent width using adaptive window selection.

    Returns
    -------
    ew, ew_err, snr, detected, best_window
    """
    if line_type == 'absorption':
        cont_window    = 100
        window_options = window_options or [5, 10, 15, 20, 30]
    else:
        cont_window    = 200
        window_options = window_options or [5, 8, 12, 10, 15, 20, 30, 50, 80, 100]

    good = (fit_mask & np.isfinite(flux) & np.isfinite(err) & (err > 0))
    cont_left   = good & (wave > line_center - cont_window - 50) & \
                         (wave < line_center - 50)
    cont_right  = good & (wave > line_center + 50) & \
                         (wave < line_center + cont_window + 50)
    cont_region = cont_left | cont_right

    if np.sum(cont_region) < 10:
        return np.nan, np.nan, 0.0, False, None

    cont_wave = wave[cont_region]
    cont_flux = flux[cont_region]
    cont_err  = err[cont_region]

    try:
        weights         = 1.0 / cont_err**2
        coeffs          = np.polyfit(cont_wave, cont_flux, deg=1, w=weights)
        continuum_model = np.poly1d(coeffs)
    except Exception:
        continuum_model = lambda w: np.median(cont_flux)

    flux_norm = flux / continuum_model(wave)
    err_norm  = err  / continuum_model(wave)

    best_snr = best_ew = best_ew_err = 0
    best_ew       = np.nan
    best_ew_err   = np.nan
    best_detected = False
    best_window   = None

    for window in window_options:
        line_region = good & (wave > line_center - window) & \
                             (wave < line_center + window)
        if np.sum(line_region) < 5:
            continue
        wave_line      = wave[line_region]
        flux_norm_line = flux_norm[line_region]
        err_norm_line  = err_norm[line_region]
        integrand      = 1 - flux_norm_line
        ew             = np.trapezoid(integrand, wave_line)
        d_lambda       = np.median(np.diff(wave_line)) if len(wave_line) > 1 else 1.0
        ew_err         = np.sqrt(np.sum(err_norm_line**2)) * d_lambda
        snr            = np.abs(ew) / ew_err if ew_err > 0 else 0.0
        detected       = snr >= min_snr
        if snr > best_snr:
            best_snr      = snr
            best_ew       = ew
            best_ew_err   = ew_err
            best_detected = detected
            best_window   = window

    return best_ew, best_ew_err, best_snr, best_detected, best_window


def compute_EW_for_all_lines(wave, flux, err, fit_mask, z):
    """Compute EW for all emission and absorption lines at redshift z."""
    results_ew = []
    all_lines  = ([(name, rest, 'emission')   for name, rest in EMISSION_LINES.items()] +
                  [(name, rest, 'absorption') for name, rest in ABSORPTION_LINES.items()])

    for name, rest, ltype in all_lines:
        obs_wave = rest * (1 + z)
        if obs_wave < wave.min() + 150 or obs_wave > wave.max() - 150:
            results_ew.append((name, obs_wave, np.nan, np.nan, 0.0, False, ltype, None))
            continue
        custom_windows = [5, 8, 10, 12] if name == 'H_alpha' else None
        ew, ew_err, snr, detected, window = \
            measure_equivalent_width_hybrid_normalized(
                wave, flux, err, fit_mask, obs_wave,
                line_type=ltype, window_options=custom_windows)
        results_ew.append((name, obs_wave, ew, ew_err, snr, detected, ltype, window))

    return results_ew


# -- Zoom inset ----------------------------------------------------------------

def make_zoom_inset(ax, spec, fit_data, best_label, comp,
                    center_waves, zoom_width, loc, line_labels,
                    x_fit, flux_resamp, err_resamp,
                    inset_width="22%", inset_height="25%",
                    connector_color='red', borderpad=1.5,
                    loc1=1, loc2=3, force_model=None,
                    show_residual_highlight=False,
                    show_ew_window=False, ew_window=None,
                    show_total_fit=True, show_connector=True,
                    ylim_percentiles=(2, 98), line_colors=None):
    """Create a zoom-in inset panel highlighting a spectral feature."""
    zoom_center = np.mean(center_waves)
    zoom_min    = zoom_center - zoom_width
    zoom_max    = zoom_center + zoom_width

    if zoom_min <= x_fit.min() or zoom_max >= x_fit.max():
        return None

    model = force_model if force_model is not None else best_label

    if model == 'Powerlaw+Galaxy' and comp.get('galpl_contrib') is not None:
        best_fit_curve = fit_data['fit_galpl']['best_fit']
        tpl_flux = comp['galpl_contrib']['tpl']
        pl_flux  = comp['galpl_contrib']['pl']
    elif model.startswith('Powerlaw+QSO') and comp.get('qsopl_contrib') is not None:
        best_fit_curve = fit_data['fit_qsopl']['best_fit']
        tpl_flux = comp['qsopl_contrib']['tpl']
        pl_flux  = comp['qsopl_contrib']['pl']
    elif model.startswith('Powerlaw+Lines') and comp.get('linepl_contrib') is not None:
        best_fit_curve = fit_data['fit_linepl']['best_fit']
        tpl_flux = comp['linepl_contrib']['tpl']
        pl_flux  = comp['linepl_contrib']['pl']
    else:
        best_fit_curve = fit_data['best_fit_params']['best_fit']
        tpl_flux = pl_flux = None

    obs_mask      = (spec['common_wave'] >= zoom_min) & (spec['common_wave'] <= zoom_max)
    fit_mask_zoom = (x_fit >= zoom_min) & (x_fit <= zoom_max)

    wave_obs = spec['common_wave'][obs_mask]
    flux_obs = flux_resamp[obs_mask]
    err_obs  = err_resamp[obs_mask]
    wave_fit = x_fit[fit_mask_zoom]
    flux_fit = best_fit_curve[fit_mask_zoom]

    if len(wave_fit) < 4:
        return None

    wave_fine  = np.linspace(wave_fit.min(), wave_fit.max(), num=20 * len(wave_fit))
    spline_tot = UnivariateSpline(wave_fit, flux_fit, k=3, s=0.5)(wave_fine)

    ax_inset = inset_axes(ax, width=inset_width, height=inset_height,
                          loc=loc, borderpad=borderpad)
    ax_inset.set_clip_on(True)
    ax_inset.plot(wave_obs, flux_obs, 'k', lw=1.2, alpha=0.7, zorder=3)
    ax_inset.fill_between(wave_obs, flux_obs - err_obs, flux_obs + err_obs,
                          color='gray', alpha=0.3, zorder=2.5)

    if show_total_fit:
        color_map = {
            'Powerlaw+Galaxy': 'red', 'Powerlaw+QSO': 'purple',
            'Powerlaw+Lines': 'cyan', 'Galaxy': 'green',
            'QSO': 'blue', 'Powerlaw': 'orange',
        }
        fit_color = next((v for k, v in color_map.items()
                          if model.startswith(k)), 'red')
        ax_inset.plot(wave_fine, spline_tot, color=fit_color, lw=2,
                      alpha=0.9, zorder=4)

    if show_residual_highlight:
        fit_on_obs    = np.interp(wave_obs, wave_fine, spline_tot)
        excess        = flux_obs - fit_on_obs
        line_center   = center_waves[0]
        highlight     = (wave_obs >= line_center - 30) & (wave_obs <= line_center + 50)
        ax_inset.fill_between(wave_obs[highlight], fit_on_obs[highlight],
                              flux_obs[highlight],
                              where=(excess[highlight] > 0),
                              color='dodgerblue', alpha=0.65, zorder=3)

    if len(flux_obs) > 0:
        y_lo  = np.nanpercentile(flux_obs, ylim_percentiles[0])
        y_hi  = np.nanpercentile(flux_obs, ylim_percentiles[1])
        y_pad = (y_hi - y_lo) * 0.25
        ax_inset.set_ylim(y_lo - y_pad, y_hi + y_pad)

    y_lim = ax_inset.get_ylim()
    for (wave, label, color) in line_labels:
        ax_inset.axvline(wave, color=color, ls=':', lw=2, alpha=0.8, zorder=5)
        ax_inset.text(wave, y_lim[1] * 0.95, label, fontsize=7,
                      ha='center', color='brown', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.9, edgecolor=color, linewidth=1.2))

    if show_ew_window and ew_window is not None:
        line_center = center_waves[0]
        ax_inset.axvspan(line_center - ew_window, line_center + ew_window,
                         color='gold', alpha=0.3)
        ax_inset.text(line_center, y_lim[0] + 0.05 * (y_lim[1] - y_lim[0]),
                      f'+/-{ew_window} A', fontsize=7, ha='center', va='bottom',
                      color='goldenrod', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.8, edgecolor='gold', linewidth=1))

    ax_inset.set_xlim(zoom_min, zoom_max)
    ax_inset.tick_params(labelsize=7, direction='in')
    ax_inset.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    if show_connector:
        patch, pp1, pp2 = mark_inset(ax, ax_inset, loc1=loc1, loc2=loc2,
                                     fc="none", ec=connector_color,
                                     linestyle='--', linewidth=1.5, alpha=0.7)
        pp1.set_clip_on(True)
        pp2.set_clip_on(True)

    return ax_inset


#-- Module-level utility functions ----------------------------------------------

def normalize_shape(p):
    """Normalise an array to its peak value for shape comparison plots."""
    return p / np.max(p) if np.max(p) > 0 else p


def get_redshift_peaks(pz_total, z_grid, min_height=0.05):
    """
    Find the top redshift solutions from the global posterior.

    Parameters
    ----------
    pz_total   : array - global p(z) posterior
    z_grid     : array - redshift grid
    min_height : float - minimum normalised height to report a peak (default 0.05)

    Returns
    -------
    peaks : list of (z, relative_height) tuples sorted by height descending
    """
    from scipy.signal import find_peaks
    p_norm      = pz_total / np.max(pz_total)
    peak_idx, _ = find_peaks(p_norm, height=min_height, distance=2, prominence=0.03)
    if len(peak_idx) == 0:
        peak_idx = [np.argmax(p_norm)]
    return sorted([(z_grid[idx], float(p_norm[idx])) for idx in peak_idx],
                  key=lambda x: x[1], reverse=True)


# -- Main plot function ----------------------------------------------------------

def plot_from_cache(results, model, save_dir=None,
                    show_fig1=False,
                    show_fig2=True,
                    show_inset=True,
                    show_residuals=True,
                    show_components=True,
                    show_line_markers=True,
                    show_fraction_label=True,
                    wavelength_range=(3600, 10400),
                    ylim=None):
    """
    Load cached results and produce diagnostic plots for one source.

    Parameters
    ----------
    results              : dict - output of RedshiftResultsCache.load_object_results()
    save_dir             : str or None - saves spectral plot as PDF if provided
    show_fig1            : bool - show chi2(z) and p(z) figure (default False)
    show_fig2            : bool - show best-fit spectrum figure (default True)
    show_inset           : bool - show zoom insets for detected lines (default True)
    show_residuals       : bool - show normalised residuals panel (default True)
    show_components      : bool - show shaded component fills (default True)
    show_line_markers    : bool - show spectral line markers (default True)
    show_fraction_label  : bool - show Host/Jet fraction text box (default True)
    model                : required. 'best', 'all', or a list of model names e.g.
                             ['Powerlaw+Galaxy', 'Powerlaw+QSO']
    wavelength_range     : tuple (wave_min, wave_max) in Angstrom
    ylim                 : tuple (y_min, y_max) or None for automatic limits

    Returns
    -------
    fig, fig2 : matplotlib Figure objects

    Examples
    --------
    fig1, fig2 = plot_from_cache(results)
    fig1, fig2 = plot_from_cache(results, show_fig1=True)
    fig1, fig2 = plot_from_cache(results, model='all')
    fig1, fig2 = plot_from_cache(results, wavelength_range=(4000, 7000))
    fig1, fig2 = plot_from_cache(results, ylim=(-2, 30))
    """
    meta   = results['metadata']
    spec   = results['spectrum']
    chi2   = results['chi2_grids']
    pz     = results['pz_results']
    lmfit  = results['lmfit_results']
    comp   = results['components']

    ID          = meta['SDSS_ID']
    obj_class   = meta['obj_class']
    z_sdss      = meta['z_sdss']
    fermi_class = meta['fermi_class']
    z_grid      = chi2['z_grid']
    best_label  = lmfit['best_label']
    z_best      = lmfit['z_best']
    z_err       = lmfit.get('z_err', np.nan)
    C_global    = pz['C_global']
    z_map_total = pz['z_map_total']

    x_fit       = spec['x_fit']
    flux_resamp = spec['flux_resamp']
    err_resamp  = spec['err_resamp']
    fit_mask    = spec['fit_mask']
    common_wave = spec['common_wave']
    fit_data    = lmfit
    best_params = lmfit['best_fit_params']

    colors_gal    = ['forestgreen', 'seagreen', 'limegreen']
    colors_qso    = plt.cm.Blues(np.linspace(0.4, 0.9, 8)).tolist()
    colors_galpl  = ['darkred', 'orangered', 'tomato']
    colors_qsopl  = colors_qso
    colors_linepl = ['cyan', 'magenta', 'yellow']

    wave_min, wave_max = wavelength_range

    # -- Figure 1: chi squared (z) and p(z) ---------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(4, 3, height_ratios=[1.6, 1.6, 1.1, 1.8],
                   hspace=0.7, wspace=0.45,
                   left=0.07, right=0.97, top=0.95, bottom=0.05)

    ax_chi    = fig.add_subplot(gs[0, :])
    ax_gal    = fig.add_subplot(gs[1, 0])
    ax_qso    = fig.add_subplot(gs[1, 1])
    ax_pl     = fig.add_subplot(gs[1, 2])
    ax_gpl    = fig.add_subplot(gs[2, 0])
    ax_qpl    = fig.add_subplot(gs[2, 1])
    ax_linepl = fig.add_subplot(gs[2, 2])
    ax_ptot   = fig.add_subplot(gs[3, :])

    def plot_pz(ax, z, pz_arr, L_parts, title):
        ax.plot(z, pz_arr, 'k-', lw=2, label='p(z)')
        for k, Li in enumerate(L_parts):
            a = np.trapezoid(Li, z)
            if a > 0:
                ax.plot(z, Li / a, lw=1.2, alpha=0.6, label=f'T{k+1}')
        if z_sdss is not None and np.isfinite(z_sdss):
            ax.axvline(z_sdss, color='gray', ls='--', lw=1.2, alpha=0.7)
        ax.set_xlabel('z (log)', fontsize=8)
        ax.set_ylabel('p(z)', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    plot_pz(ax_gal,    z_grid, pz['pz_gal'],    pz['L_gal_parts'],    'Galaxy p(z)')
    plot_pz(ax_qso,    z_grid, pz['pz_qso'],    pz['L_qso_parts'],    'QSO p(z)')
    plot_pz(ax_pl,     z_grid, pz['pz_pl'],     pz['L_pl_parts'],     'PL p(z)')
    plot_pz(ax_gpl,    z_grid, pz['pz_galpl'],  pz['L_galpl_parts'],  'PL+Galaxy p(z)')
    plot_pz(ax_qpl,    z_grid, pz['pz_qsopl'],  pz['L_qsopl_parts'],  'PL+QSO p(z)')
    plot_pz(ax_linepl, z_grid, pz['pz_linepl'], pz['L_linepl_parts'], 'PL+Lines p(z)')

    def normalize_shape(p):
        return p / np.max(p) if np.max(p) > 0 else p

    for arr, color, label in [
        (pz['pz_gal'],    'g',      'Galaxy'),
        (pz['pz_qso'],    'b',      'QSO'),
        (pz['pz_galpl'],  'r',      'PL+Gal'),
        (pz['pz_qsopl'],  'm',      'PL+QSO'),
        (pz['pz_pl'],     'orange', 'PL'),
        (pz['pz_linepl'], 'cyan',   'PL+Lines'),
    ]:
        ax_ptot.plot(z_grid, normalize_shape(arr), color=color,
                     lw=1.6, alpha=0.3, label=label)

    ax_ptot.plot(z_grid, pz['p_total'], 'k-', lw=2.5, label='Global p(z)')
    ax_ptot.axvline(z_map_total, color='red', ls=':', lw=1.4, alpha=0.7,
                    label=f'z_MAP={z_map_total:.3f}')
    if z_sdss is not None and np.isfinite(z_sdss):
        ax_ptot.axvline(z_sdss, color='gray', ls='--', lw=1.2,
                        alpha=0.7, label='SDSS z')
    ax_ptot.set_xlabel('Redshift z (log)', fontsize=9)
    ax_ptot.set_ylabel('Probability density', fontsize=9)
    ax_ptot.set_title(f'Global p(z) - C_global={C_global:.1f}', fontsize=10)
    ax_ptot.grid(alpha=0.3)
    ax_ptot.set_xscale('log')
    ax_ptot.legend(fontsize=7, ncol=3, loc='upper right')

    def plot_chi2(ax, z, chi, **kw):
        m = np.isfinite(chi) & (chi > 0)
        if np.any(m):
            ax.plot(z[m], chi[m], **kw)

    for k, c in enumerate(chi2['chi2_gal_list']):
        plot_chi2(ax_chi, z_grid, c,
                  color=colors_gal[k % len(colors_gal)], lw=1.5,
                  label=f'Gal T{k+1}')
    for k, c in enumerate(chi2['chi2_qso_list']):
        plot_chi2(ax_chi, z_grid, c,
                  color=colors_qso[k % len(colors_qso)], lw=1.5, ls='--',
                  label=f'QSO T{k+1}')
    for k, c in enumerate(chi2['chi2_galpl_list']):
        plot_chi2(ax_chi, z_grid, c,
                  color=colors_galpl[k % len(colors_galpl)], lw=1.5,
                  label=f'PL+Gal T{k+1}')
    for k, c in enumerate(chi2['chi2_qsopl_list']):
        plot_chi2(ax_chi, z_grid, c,
                  color=colors_qsopl[k % len(colors_qsopl)], lw=1.5, ls='-.',
                  label=f'PL+QSO T{k+1}')
    for k, c in enumerate(chi2['chi2_linepl_list']):
        plot_chi2(ax_chi, z_grid, c,
                  color=colors_linepl[k % len(colors_linepl)], lw=1.5,
                  ls=(0, (3, 1, 1, 1)), label=f'PL+Line T{k+1}')
    if chi2['chi2_pl'] is not None:
        plot_chi2(ax_chi, z_grid, chi2['chi2_pl'],
                  color='orange', lw=1.5, ls=':', label='Powerlaw')

    if z_sdss is not None and np.isfinite(z_sdss):
        ax_chi.axvline(z_sdss, color='gray', ls='--', alpha=0.7,
                       lw=1.5, label='SDSS z')
    ax_chi.axvline(z_map_total, color='red', ls=':', lw=1.5, alpha=0.8,
                   label=f'z_MAP={z_map_total:.3f}')
    ax_chi.set_xscale('log')
    ax_chi.set_xlabel('Redshift z (log)', fontsize=9)
    ax_chi.set_ylabel(r'$\chi^2$', fontsize=9)
    ax_chi.set_title(f'SDSS_ID: {ID} - $\\chi^2(z)$ for all templates',
                     fontsize=10)
    ax_chi.grid(alpha=0.3)
    ax_chi.legend(ncol=3, fontsize=6)

    plt.tight_layout()
    if show_fig1:
        plt.show()
    else:
        plt.close(fig)
        fig = None

    # -- Figure 2: Best-fit spectrum --------------------------------------------
    if not show_fig2:
        return fig, None

    if show_residuals:
        fig2, axs2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [4, 1],
                                                'hspace': 0.05})
    else:
        fig2, ax_only = plt.subplots(1, 1, figsize=(14, 6))
        axs2 = [ax_only]

    axs2[0].plot(common_wave, flux_resamp, 'k', lw=1, alpha=0.7,
                 label='Observed Spectrum')
    axs2[0].fill_between(common_wave,
                          flux_resamp - err_resamp,
                          flux_resamp + err_resamp,
                          color='gray', alpha=0.4, label=r'$\pm 1\sigma$')

    fit_curves = [
        ('fit_gal',    'g',      1.5, {'alpha': 0.5},                'Galaxy Fit'),
        ('fit_qso',    'b',      1.5, {'alpha': 0.5},                'QSO Fit'),
        ('fit_pl',     'orange', 1.5, {'ls': '--', 'alpha': 0.5},    'Powerlaw Fit'),
        ('fit_qsopl',  'purple', 1.5, {'alpha': 0.5},                'Powerlaw+QSO'),
        ('fit_linepl', 'cyan',   1.5, {'alpha': 0.5},                'Powerlaw+Lines'),
        ('fit_galpl',  'r',      1.5, {'alpha': 0.5},                'Powerlaw+Galaxy'),
    ]
    key_to_family = {
        'fit_gal'    : 'Galaxy',
        'fit_qso'    : 'QSO',
        'fit_pl'     : 'Powerlaw',
        'fit_galpl'  : 'Powerlaw+Galaxy',
        'fit_qsopl'  : 'Powerlaw+QSO',
        'fit_linepl' : 'Powerlaw+Lines',
    }

    def _is_best_model(key, best_label, key_to_family):
        """Check if this key corresponds to the best-fit model."""
        family = key_to_family.get(key, '')
        if family == 'Powerlaw':
            return best_label == 'Powerlaw'
        return best_label.startswith(family)

    for key, color, lw, kwargs, label in fit_curves:
        family = key_to_family.get(key, '')

        if model == 'best':
            # Show only the exact best-fit model
            if not _is_best_model(key, best_label, key_to_family):
                continue

        elif model == 'all':
            # Show everything — no filter
            pass

        elif isinstance(model, list):
            # List can contain: 'best', model family names, or label strings
            show = False
            for m in model:
                if m == 'best':
                    # 'best' in list means: include the best-fit model
                    if _is_best_model(key, best_label, key_to_family):
                        show = True
                elif label == m or label.replace(' Fit', '') == m or family == m:
                    show = True
            if not show:
                continue

        else:
            # Single string model name e.g. model='Powerlaw+Galaxy'
            if label != model and label.replace(' Fit', '') != model and family != model:
                continue

        fd = fit_data.get(key)
        if fd is not None and fd.get('best_fit') is not None:
            is_best    = _is_best_model(key, best_label, key_to_family)
            plot_lw    = 3.0 if is_best else lw
            plot_alpha = 1.0 if is_best else kwargs.get('alpha', 0.5)
            plot_kwargs = {k: v for k, v in kwargs.items() if k != 'alpha'}
            axs2[0].plot(x_fit, fd['best_fit'], color=color,
                         lw=plot_lw, alpha=plot_alpha,
                         label=label, **plot_kwargs)

    # y-limits
    all_flux = [flux_resamp]
    if best_label == 'Powerlaw+Galaxy' and comp['galpl_contrib'] is not None:
        all_flux += [comp['galpl_contrib']['tpl'], comp['galpl_contrib']['pl']]
    elif best_label.startswith('Powerlaw+QSO') and comp['qsopl_contrib'] is not None:
        all_flux += [comp['qsopl_contrib']['tpl'], comp['qsopl_contrib']['pl']]
    elif best_label.startswith('Powerlaw+Lines') and comp['linepl_contrib'] is not None:
        all_flux += [comp['linepl_contrib']['tpl'], comp['linepl_contrib']['pl']]

    combined = np.concatenate([f[np.isfinite(f)] for f in all_flux])
    if ylim is not None:
        axs2[0].set_ylim(ylim[0], ylim[1])
    else:
        y_lo  = np.nanpercentile(combined, 1)
        y_hi  = np.nanpercentile(flux_resamp, 99)
        y_pad = (y_hi - y_lo) * 0.25
        axs2[0].set_ylim(y_lo - y_pad, y_hi + y_pad)
    axs2[0].set_xlim(wave_min, wave_max)

    # Line markers - must come before zoom insets
    results_ew = compute_EW_for_all_lines(
        common_wave, flux_resamp, err_resamp, fit_mask, z_best)
    for name, obs, ew, ew_err, snr, detected, ltype, window in results_ew:
        if detected:
            print(f"  {name:15s}: EW = {ew:6.2f} ± {ew_err:5.2f} Å, "
                  f"SNR = {snr:4.1f}")
    if show_line_markers:
        add_line_markers_to_plot(axs2[0], z_best,
                                  x_range=(wave_min, wave_max))

    # Component shading
    if show_components and best_label == 'Powerlaw+Galaxy' and comp['galpl_contrib'] is not None:
        axs2[0].fill_between(x_fit, comp['galpl_contrib']['tpl'],
                              color='green', alpha=0.35, label='Host contrib.')
        axs2[0].fill_between(x_fit, comp['galpl_contrib']['pl'],
                              color='orange', alpha=0.40, label='Jet contrib.')
        frac_gal = comp['galpl_contrib']['frac_tpl']
        frac_jet = comp['galpl_contrib']['frac_pl']
        if show_fraction_label:
         axs2[0].text(0.02, 0.98,
                     f"Host: {frac_gal*100:.1f}%\nJet: {frac_jet*100:.1f}%",
                     transform=axs2[0].transAxes, va='top', fontsize=12,
                     weight='bold',
                     bbox=dict(boxstyle='round', facecolor='wheat',
                               alpha=0.9, edgecolor='black', linewidth=2.5))

    elif best_label.startswith('Powerlaw+QSO') and comp['qsopl_contrib'] is not None:
        axs2[0].fill_between(x_fit, comp['qsopl_contrib']['tpl'],
                              color='blue', alpha=0.35, label='Disk/BLR contrib.')
        axs2[0].fill_between(x_fit, comp['qsopl_contrib']['pl'],
                              color='orange', alpha=0.40, label='Jet contrib.')
        frac_qso = comp['qsopl_contrib']['frac_tpl']
        frac_jet = comp['qsopl_contrib']['frac_pl']
        if show_fraction_label:
         axs2[0].text(0.02, 0.98,
                     f"Disk/BLR: {frac_qso*100:.1f}%\nJet: {frac_jet*100:.1f}%",
                     transform=axs2[0].transAxes, va='top', fontsize=12,
                     weight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue',
                               alpha=0.9, edgecolor='black', linewidth=2.5))

    elif best_label.startswith('Powerlaw+Lines') and comp['linepl_contrib'] is not None:
        axs2[0].fill_between(x_fit, comp['linepl_contrib']['tpl'],
                              color='cyan', alpha=0.35, label='Disk/BLR contrib.')
        axs2[0].fill_between(x_fit, comp['linepl_contrib']['pl'],
                              color='orange', alpha=0.40, label='Jet contrib.')
        frac_line = comp['linepl_contrib']['frac_tpl']
        frac_jet  = comp['linepl_contrib']['frac_pl']
        if show_fraction_label:
         axs2[0].text(0.02, 0.98,
                     f"Disk/BLR: {frac_line*100:.1f}%\nJet: {frac_jet*100:.1f}%",
                     transform=axs2[0].transAxes, va='top', fontsize=12,
                     weight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightcyan',
                               alpha=0.9, edgecolor='black', linewidth=2.5))

    # Zoom insets -- only for detected lines when show_inset=True
    if show_inset:
        cah_obs = 3967.5 * (1 + z_best)
        for name, obs_wave, ew, ew_err, snr, detected, ltype, window in results_ew:
            if not detected:
                continue
            if name == 'Ca II K' and snr >= 5:
                make_zoom_inset(
                    ax=axs2[0], spec=spec, fit_data=fit_data,
                    best_label=best_label, comp=comp,
                    center_waves=[obs_wave, cah_obs], zoom_width=140,
                    loc='lower right',
                    line_labels=[(obs_wave, 'Ca II K', 'forestgreen'),
                                 (cah_obs, 'Ca II H', 'forestgreen')],
                    x_fit=x_fit, flux_resamp=flux_resamp,
                    err_resamp=err_resamp, line_colors={},
                    inset_width="25%", inset_height="35%",
                    connector_color='red', loc1=1, loc2=3)
            elif name == 'Mg II':
                make_zoom_inset(
                    ax=axs2[0], spec=spec, fit_data=fit_data,
                    best_label=best_label, comp=comp,
                    center_waves=[obs_wave], zoom_width=150,
                    loc='lower left',
                    line_labels=[(obs_wave, 'Mg II', 'darkorange')],
                    x_fit=x_fit, flux_resamp=flux_resamp,
                    err_resamp=err_resamp, line_colors={},
                    inset_width="22%", inset_height="30%",
                    connector_color='darkorange', loc1=2, loc2=4,
                    show_residual_highlight=False, show_total_fit=False,
                    show_ew_window=True, ew_window=window,
                    show_connector=True, ylim_percentiles=(5, 99))
            elif name == 'H_alpha':
                make_zoom_inset(
                    ax=axs2[0], spec=spec, fit_data=fit_data,
                    best_label=best_label, comp=comp,
                    center_waves=[obs_wave], zoom_width=150,
                    loc='upper right',
                    line_labels=[(obs_wave, r'H$\alpha$', 'red')],
                    x_fit=x_fit, flux_resamp=flux_resamp,
                    err_resamp=err_resamp, line_colors={},
                    inset_width="22%", inset_height="30%",
                    connector_color='red', loc1=3, loc2=4,
                    show_residual_highlight=True, show_total_fit=False,
                    show_connector=True, ylim_percentiles=(5, 99.5))

    # Labels
    z_sdss_str = (f"{z_sdss:.4f}"
                  if z_sdss is not None and np.isfinite(z_sdss) else "N/A")
    z_err_str  = f" ± {z_err:.4f}" if np.isfinite(z_err) else ""

    axs2[0].set_ylabel(r'Flux [$10^{-17}$ erg/s/cm$^2$/Å]',
                        fontsize=11, fontweight='bold')
    axs2[0].set_title(
        f'SDSS_ID: {ID} | SDSS CLASS: {obj_class} | 4FGL CLASS: {fermi_class}\n'
        f'Best model: {best_label} | '
        f'z_fit={z_best:.4f}{z_err_str} | z_SDSS={z_sdss_str}',
        fontsize=12, fontweight='bold')
    axs2[0].legend(fontsize=10, ncol=2, framealpha=0.95)
    axs2[0].grid(alpha=0.3)

    # Residuals
    if show_residuals and len(axs2) > 1:
        if best_params is not None and best_params['best_fit'] is not None:
            resi = (spec['y_fit'] - best_params['best_fit']) / spec['e_fit']
            axs2[1].plot(x_fit, resi, color='purple', lw=1,
                         label='Normalized residuals')
            axs2[1].axhline(0, color='gray', ls='--', lw=1)
            axs2[1].fill_between(x_fit, -3, 3, color='gray', alpha=0.2,
                                  label=r'$\pm3\sigma$')
        axs2[1].set_ylim(-5, 5)
        axs2[1].set_xlabel(r'Wavelength [A]', fontsize=12, fontweight='bold')
        axs2[1].set_ylabel(r'Residual (sigma)', fontsize=12, fontweight='bold')
        axs2[1].legend(fontsize=10)
        axs2[1].grid(alpha=0.3)
    else:
        axs2[0].set_xlabel(r'Wavelength [A]', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Save
    if save_dir is not None:
        outdir = Path(save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig2.savefig(outdir / f"{int(ID)}.pdf", format='pdf',
                     bbox_inches='tight')
        print(f"Saved: {outdir / f'{int(ID)}.pdf'}")

    print(f"\nSDSS_ID: {ID}")
    print(f"  z_MAP={z_map_total:.4f}, z_fit={z_best:.4f}{z_err_str}, "
          f"z_SDSS={z_sdss_str}")
    print(f"  Best model: {best_label} "
          f"(chi2={best_params['chisqr']:.2f})")

    return fig, fig2


# -- Multi-object comparison plot ----------------------------------------------

def plot_multi_object_comparison_single(object_list, cache, final_bl,
                                        n_cols=2, save_dir="paper_plots"):
    """
    Create a multi-panel comparison plot for a list of sources.

    Parameters
    ----------
    object_list : list of dicts with keys 'SDSS_ID' and optionally 'MJD'
    cache       : RedshiftResultsCache instance
    final_bl    : astropy Table - the main blazar sample table
    n_cols      : int - number of columns in the grid
    save_dir    : str - directory to save the output PDF
    """
    import string
    n_objects   = len(object_list)
    n_rows      = int(np.ceil(n_objects / n_cols))
    panel_labels = list(string.ascii_lowercase[:n_objects])
    WAVE_MIN, WAVE_MAX = 3800, 10000

    fig = plt.figure(figsize=(11 * n_cols, 8 * n_rows))

    left_margin  = 0.07;  right_margin = 0.02
    top_margin   = 0.02;  bottom_margin = 0.05
    col_spacing  = 0.03;  row_spacing  = 0.04
    spec_resid_gap = 0.01

    usable_width     = 1.0 - left_margin - right_margin
    usable_height    = 1.0 - top_margin  - bottom_margin
    panel_width      = (usable_width  - (n_cols - 1) * col_spacing) / n_cols
    total_row_height = (usable_height - (n_rows - 1) * row_spacing) / n_rows
    spec_height      = total_row_height * 0.78
    resid_height     = total_row_height * 0.20

    for idx, obj in enumerate(object_list):
        row_idx = idx // n_cols
        col_idx = idx  % n_cols

        left         = left_margin + col_idx * (panel_width + col_spacing)
        top_of_obj   = 1.0 - top_margin - row_idx * (total_row_height + row_spacing)
        spec_bottom  = top_of_obj - spec_height
        resid_bottom = spec_bottom - spec_resid_gap - resid_height

        ax_spec  = fig.add_axes([left, spec_bottom,  panel_width, spec_height])
        ax_resid = fig.add_axes([left, resid_bottom, panel_width, resid_height],
                                sharex=ax_spec)

        sdss_id = obj['SDSS_ID']
        mjd     = obj.get('MJD', None)

        try:
            results = cache.load_object_results(str(sdss_id), mjd)
        except Exception as e:
            print(f"  Failed to load {sdss_id}: {e}")
            continue

        meta      = results['metadata']
        spec      = results['spectrum']
        lmfit_res = results['lmfit_results']
        comp      = results['components']

        row_match  = final_bl[final_bl['SDSS_ID'] == sdss_id]
        fhl_class  = row_match['CLASS1'][0]       if len(row_match) > 0 else "N/A"
        sdss_name  = row_match['SDSS_NAME'][0]    if len(row_match) > 0 else "N/A"
        daic       = row_match['aicc_margin'][0]  if len(row_match) > 0 else np.nan
        RCHI       = row_match['RCHI2'][0]        if len(row_match) > 0 else np.nan
        sn         = row_match['SN_MEDIAN_ALL'][0]if len(row_match) > 0 else np.nan
        lm_rchi    = row_match['R_lm_chi'][0]     if len(row_match) > 0 else np.nan
        fhl_z      = row_match['fhl_z'][0]        if len(row_match) > 0 else np.nan

        z_best     = lmfit_res['z_best']
        z_sdss     = meta['z_sdss']
        best_label = lmfit_res['best_label']
        obj_class  = meta['obj_class']
        best_params= lmfit_res['best_fit_params']
        z_err      = lmfit_res.get('z_err', np.nan)
        fit_data   = lmfit_res

        x_fit       = spec['x_fit']
        flux_resamp = spec['flux_resamp']
        common_wave = spec['common_wave']
        err_resamp  = spec['err_resamp']
        fit_mask    = spec['fit_mask']

        wave_mask  = (common_wave >= WAVE_MIN) & (common_wave <= WAVE_MAX)
        xfit_mask  = (x_fit      >= WAVE_MIN) & (x_fit      <= WAVE_MAX)
        wave_plot  = common_wave[wave_mask]
        flux_plot  = flux_resamp[wave_mask]
        err_plot   = err_resamp[wave_mask]
        x_fit_plot = x_fit[xfit_mask]

        ax_spec.plot(wave_plot, flux_plot, 'k', lw=1, alpha=0.7, label='Observed')
        ax_spec.fill_between(wave_plot, flux_plot - err_plot,
                             flux_plot + err_plot,
                             color='gray', alpha=0.4, label=r'$\pm1\sigma$')

        is_pl_gal   = (best_label == 'Powerlaw+Galaxy')
        is_pl_qso   = best_label.startswith('Powerlaw+QSO')
        is_pl_lines = best_label.startswith('Powerlaw+Lines')
        is_galaxy   = (best_label == 'Galaxy')
        is_qso      = ('QSO' in best_label and 'Powerlaw' not in best_label)
        is_pl_only  = (best_label == 'Powerlaw')

        if is_pl_gal and fit_data['fit_galpl'] is not None:
            ax_spec.plot(x_fit_plot,
                         fit_data['fit_galpl']['best_fit'][xfit_mask],
                         'r', lw=2.5, label='PL+Gal')
            if comp['galpl_contrib'] is not None:
                ax_spec.fill_between(x_fit_plot,
                                     comp['galpl_contrib']['tpl'][xfit_mask],
                                     color='green', alpha=0.55, label='Host')
                ax_spec.fill_between(x_fit_plot,
                                     comp['galpl_contrib']['pl'][xfit_mask],
                                     color='orange', alpha=0.55, label='Jet')
                frac_gal = comp['galpl_contrib']['frac_tpl']
                frac_jet = comp['galpl_contrib']['frac_pl']
                ax_spec.text(0.02, 0.98,
                             f"Host: {frac_gal*100:.0f}%\nJet: {frac_jet*100:.0f}%",
                             transform=ax_spec.transAxes, va='top',
                             fontsize=12, weight='bold',
                             bbox=dict(boxstyle='round', facecolor='wheat',
                                       alpha=0.85, edgecolor='black'))

        elif is_pl_qso and fit_data['fit_qsopl'] is not None:
            ax_spec.plot(x_fit_plot,
                         fit_data['fit_qsopl']['best_fit'][xfit_mask],
                         'purple', lw=2.5, label='PL+QSO')
            if comp['qsopl_contrib'] is not None:
                ax_spec.fill_between(x_fit_plot,
                                     comp['qsopl_contrib']['tpl'][xfit_mask],
                                     color='blue', alpha=0.55, label='Disk/BLR')
                ax_spec.fill_between(x_fit_plot,
                                     comp['qsopl_contrib']['pl'][xfit_mask],
                                     color='orange', alpha=0.55, label='Jet')
                frac_qso = comp['qsopl_contrib']['frac_tpl']
                frac_jet = comp['qsopl_contrib']['frac_pl']
                ax_spec.text(0.02, 0.98,
                             f"Disk/BLR: {frac_qso*100:.0f}%\n"
                             f"Jet: {frac_jet*100:.0f}%",
                             transform=ax_spec.transAxes, va='top',
                             fontsize=12, weight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightblue',
                                       alpha=0.85, edgecolor='black'))

        elif is_pl_lines and fit_data['fit_linepl'] is not None:
            ax_spec.plot(x_fit_plot,
                         fit_data['fit_linepl']['best_fit'][xfit_mask],
                         'cyan', lw=2.5, label='PL+Lines')
            if comp['linepl_contrib'] is not None:
                ax_spec.fill_between(x_fit_plot,
                                     comp['linepl_contrib']['tpl'][xfit_mask],
                                     color='cyan', alpha=0.55, label='Lines')
                ax_spec.fill_between(x_fit_plot,
                                     comp['linepl_contrib']['pl'][xfit_mask],
                                     color='orange', alpha=0.55, label='Jet')
                frac_line = comp['linepl_contrib']['frac_tpl']
                frac_jet  = comp['linepl_contrib']['frac_pl']
                ax_spec.text(0.02, 0.98,
                             f"BLR: {frac_line*100:.0f}%\n"
                             f"Jet: {frac_jet*100:.0f}%",
                             transform=ax_spec.transAxes, va='top',
                             fontsize=12, weight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightcyan',
                                       alpha=0.85, edgecolor='black'))

        elif is_galaxy and fit_data['fit_gal'] is not None:
            ax_spec.plot(x_fit_plot,
                         fit_data['fit_gal']['best_fit'][xfit_mask],
                         'g', lw=2.5, label='Galaxy')

        elif is_qso and fit_data['fit_qso'] is not None:
            ax_spec.plot(x_fit_plot,
                         fit_data['fit_qso']['best_fit'][xfit_mask],
                         'b', lw=2.5, label='QSO')

        elif is_pl_only and fit_data['fit_pl'] is not None:
            ax_spec.plot(x_fit_plot,
                         fit_data['fit_pl']['best_fit'][xfit_mask],
                         'orange', lw=2.5, label='PL')

        # y-limits
        all_flux = [flux_plot]
        if is_pl_gal and comp['galpl_contrib'] is not None:
            all_flux += [comp['galpl_contrib']['tpl'][xfit_mask],
                         comp['galpl_contrib']['pl'][xfit_mask]]
        elif is_pl_qso and comp['qsopl_contrib'] is not None:
            all_flux += [comp['qsopl_contrib']['tpl'][xfit_mask],
                         comp['qsopl_contrib']['pl'][xfit_mask]]
        elif is_pl_lines and comp['linepl_contrib'] is not None:
            all_flux += [comp['linepl_contrib']['tpl'][xfit_mask],
                         comp['linepl_contrib']['pl'][xfit_mask]]

        combined = np.concatenate([f[np.isfinite(f)] for f in all_flux])
        y_lo  = np.nanpercentile(combined,  1)
        y_hi  = np.nanpercentile(flux_plot, 99)
        y_pad = (y_hi - y_lo) * 0.25
        ax_spec.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # Line markers
        add_line_markers_to_plot(ax_spec, z_best,
                                  x_range=(WAVE_MIN, WAVE_MAX))

        # Zoom inset
        cak_obs  = 3933.7 * (1 + z_best)
        cah_obs  = 3967.5 * (1 + z_best)
        mgii_obs = 2798.0 * (1 + z_best)
        hb_obs   = 4861.3 * (1 + z_best)
        ha_obs   = 6562.8 * (1 + z_best)
        civ_obs  = 1549.0 * (1 + z_best)

        zoom_center = None
        if is_pl_gal:
            frac_gal = (comp['galpl_contrib']['frac_tpl']
                        if comp['galpl_contrib'] is not None else 0.0)
            if frac_gal >= 0.01:
                zoom_center = [cak_obs, cah_obs]
                zoom_width  = 150
                zoom_lines  = [(cak_obs, 'Ca II K', 'forestgreen'),
                               (cah_obs, 'Ca II H', 'forestgreen')]
                zoom_loc    = 'lower center'
                loc1, loc2  = 1, 3

        elif is_pl_qso or is_pl_lines or is_qso:
            for obs, label, color in [
                (mgii_obs, 'Mg II', 'dodgerblue'),
                (civ_obs,  'C IV',  'dodgerblue'),
                (hb_obs,   'H_beta','dodgerblue'),
                (ha_obs,   'H_alpha','forestgreen'),
            ]:
                if WAVE_MIN <= obs <= WAVE_MAX:
                    zoom_center = [obs]
                    zoom_width  = 150
                    zoom_lines  = [(obs, label, color)]
                    zoom_loc    = 'lower right'
                    loc1, loc2  = 1, 3
                    break

        elif is_galaxy:
            zoom_center = [cak_obs, cah_obs]
            zoom_width  = 150
            zoom_lines  = [(cak_obs, 'Ca II K', 'forestgreen'),
                           (cah_obs, 'Ca II H', 'forestgreen')]
            zoom_loc    = 'lower right'
            loc1, loc2  = 1, 3

        if zoom_center is not None:
            zoom_mean = np.mean(zoom_center)
            if WAVE_MIN + zoom_width / 2 < zoom_mean < WAVE_MAX - zoom_width / 2:
                make_zoom_inset(
                    ax=ax_spec, spec=spec, fit_data=fit_data,
                    best_label=best_label, comp=comp,
                    center_waves=zoom_center, zoom_width=zoom_width,
                    loc=zoom_loc, line_labels=zoom_lines, line_colors={},
                    x_fit=x_fit, flux_resamp=flux_resamp,
                    err_resamp=err_resamp,
                    inset_width="30%", inset_height="35%",
                    connector_color='red', borderpad=1.0,
                    loc1=loc1, loc2=loc2)

        # Title and labels
        z_sdss_str = (f"{z_sdss:.3f}"
                      if z_sdss is not None and np.isfinite(z_sdss) else "N/A")
        z_err_str  = f"±{z_err:.3f}" if not np.isnan(z_err) else ""

        ax_spec.set_title(
            f"({panel_labels[idx]}) {sdss_name} | {fhl_class} | "
            f"SDSS: {obj_class} (z={z_sdss_str}) | fhl_z={fhl_z:.3f}\n"
            f"Best: {best_label} | z_fit={z_best:.3f}{z_err_str} | "
            f"χ²_r SDSS={RCHI:.2f} lmfit={lm_rchi:.2f} | "
            f"ΔAICc={daic:.1f} | S/N={sn:.1f}",
            fontsize=11, fontweight='bold', pad=2)

        ax_spec.set_ylabel(r'Flux [$10^{-17}$ erg/s/cm$^2$/Å]',
                            fontsize=12, fontweight='bold')
        ax_spec.tick_params(labelsize=9, labelbottom=False)
        ax_spec.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax_spec.grid(alpha=0.3)
        ax_spec.set_xlim(WAVE_MIN, WAVE_MAX)

        # Residuals
        if (best_params is not None and
                best_params.get('best_fit') is not None):
            resi = ((spec['y_fit'] - best_params['best_fit'])
                    / spec['e_fit'])
            ax_resid.plot(x_fit[xfit_mask], resi[xfit_mask],
                          color='purple', lw=1)
            ax_resid.axhline(0, color='gray', ls='--', lw=1.2)
            ax_resid.fill_between(x_fit[xfit_mask], -3, 3,
                                  color='gray', alpha=0.2)
            rms = np.sqrt(np.mean(resi[xfit_mask]**2))
            ax_resid.text(0.98, 0.95, f'RMS={rms:.2f}σ',
                          transform=ax_resid.transAxes,
                          va='top', ha='right', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='white',
                                    alpha=0.8, edgecolor='black'))

        ax_resid.set_ylim(-5, 5)
        ax_resid.set_xlim(WAVE_MIN, WAVE_MAX)
        ax_resid.set_xlabel(r'Wavelength [Å]', fontsize=12, fontweight='bold')
        ax_resid.set_ylabel(r'Resid. ($\sigma$)', fontsize=12, fontweight='bold')
        ax_resid.tick_params(labelsize=9)
        ax_resid.grid(alpha=0.3)

        print(f"  [{idx+1}] {best_label}, z={z_best:.3f}")

    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, "multi_object_comparison.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.show()
    return fig


# -- New utility functions ------------------------------------------------------

def classification_summary(results):
    """
    Print a one-line human-readable classification summary for a source.

    Parameters
    ----------
    results : dict - output of RedshiftResultsCache.load_object_results()

    Returns
    -------
    summary : str
    """
    meta   = results['metadata']
    lmfit  = results['lmfit_results']
    comp   = results['components']
    spec   = results['spectrum']

    best   = lmfit['best_label']
    z      = lmfit['z_best']
    sn     = meta['sn_median']

    if best == 'Powerlaw+Galaxy' or best == 'Powerlaw':
        blazar_class = 'BL Lac candidate'
    elif best.startswith('Powerlaw+QSO') or best.startswith('Powerlaw+Lines'):
        blazar_class = 'FSRQ candidate'
    elif best == 'Galaxy':
        blazar_class = 'Host-galaxy dominated'
    elif best == 'QSO':
        blazar_class = 'QSO-dominated'
    else:
        blazar_class = 'Unclassified'

    jet_str = ''
    if best == 'Powerlaw+Galaxy' and comp['galpl_contrib'] is not None:
        fjet = comp['galpl_contrib']['frac_pl']
        jet_str = f'Jet-dominated ({fjet*100:.1f}%) | Host galaxy detected'
    elif best.startswith('Powerlaw+QSO') and comp['qsopl_contrib'] is not None:
        fjet = comp['qsopl_contrib']['frac_pl']
        jet_str = f'Jet-dominated ({fjet*100:.1f}%) | Disc/BLR detected'
    elif best.startswith('Powerlaw+Lines') and comp['linepl_contrib'] is not None:
        fjet = comp['linepl_contrib']['frac_pl']
        jet_str = f'Jet-dominated ({fjet*100:.1f}%) | Emission lines only'
    elif best == 'Powerlaw':
        jet_str = 'Featureless jet continuum | No host detected'

    results_ew = compute_EW_for_all_lines(
        spec['common_wave'], spec['flux_resamp'],
        spec['err_resamp'], spec['fit_mask'], z)
    detected_lines = [
        f"{name} ({ltype[:3]}, S/N={snr:.1f})"
        for name, obs, ew, ew_err, snr, detected, ltype, window in results_ew
        if detected
    ]
    lines_str = ' | '.join(detected_lines[:3]) if detected_lines else 'No lines detected'

    summary = (f"{blazar_class} at z={z:.4f} | {jet_str} | "
               f"S/N={sn:.1f} | {lines_str}")

    print(f"\n{'='*70}")
    print(f"SDSS_ID: {meta['SDSS_ID']} | Fermi: {meta['fermi_class']} | "
          f"SDSS: {meta['obj_class']}")
    print(f"{'-'*70}")
    print(summary)
    print(f"{'='*70}\n")
    return summary


def compare_epochs(sdss_id, mjd_list, cache,
                   wavelength_range=(3600, 10400),
                   show_model=True,
                   save_dir=None):
    """
    Overlay spectra from multiple epochs for a single source.

    Parameters
    ----------
    sdss_id          : str
    mjd_list         : list of int
    cache            : RedshiftResultsCache instance
    wavelength_range : tuple (wave_min, wave_max)
    show_model       : bool - overlay best-fit model
    save_dir         : str or None
    """
    wave_min, wave_max = wavelength_range
    n_epochs = len(mjd_list)
    cmap     = plt.cm.viridis
    colors   = cmap(np.linspace(0.15, 0.95, n_epochs))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})

    print(f"\nComparing {n_epochs} epochs for SDSS_ID {sdss_id}")
    print("="*60)

    epoch_data = []
    for j, mjd in enumerate(sorted(mjd_list)):
        try:
            res   = cache.load_object_results(str(sdss_id), mjd=mjd)
            spec  = res['spectrum']
            lmfit = res['lmfit_results']
            comp  = res['components']
            wave  = spec['common_wave']
            flux  = spec['flux_resamp']
            err   = spec['err_resamp']
            z     = lmfit['z_best']
            best  = lmfit['best_label']
            mask  = (wave >= wave_min) & (wave <= wave_max)

            jet_frac = np.nan
            if best == 'Powerlaw+Galaxy' and comp['galpl_contrib'] is not None:
                jet_frac = comp['galpl_contrib']['frac_pl']
            elif best.startswith('Powerlaw+QSO') and comp['qsopl_contrib'] is not None:
                jet_frac = comp['qsopl_contrib']['frac_pl']
            elif best.startswith('Powerlaw+Lines') and comp['linepl_contrib'] is not None:
                jet_frac = comp['linepl_contrib']['frac_pl']

            jf_str = f" | Jet={jet_frac*100:.1f}%" if np.isfinite(jet_frac) else ""
            label  = f"MJD={mjd} | z={z:.4f} | {best.split('(')[0].strip()}{jf_str}"

            axes[0].plot(wave[mask], flux[mask], color=colors[j], lw=1.2,
                         alpha=0.8, label=label)
            axes[0].fill_between(wave[mask], flux[mask]-err[mask],
                                  flux[mask]+err[mask],
                                  color=colors[j], alpha=0.15)

            if show_model and lmfit['best_fit_params']['best_fit'] is not None:
                x_fit    = spec['x_fit']
                best_fit = lmfit['best_fit_params']['best_fit']
                xm       = (x_fit >= wave_min) & (x_fit <= wave_max)
                axes[0].plot(x_fit[xm], best_fit[xm], color=colors[j],
                             lw=2.5, ls='--', alpha=0.9)

            epoch_data.append({'mjd': mjd, 'wave': wave[mask],
                                'flux': flux[mask], 'z': z, 'color': colors[j]})
            print(f"  MJD={mjd:<8} z={z:.4f}  {best}{jf_str}")

        except Exception as e:
            print(f"  MJD={mjd}: Failed - {e}")

    if len(epoch_data) > 1:
        ref_wave = epoch_data[0]['wave']
        ref_flux = epoch_data[0]['flux']
        for ep in epoch_data[1:]:
            ratio = np.interp(ref_wave, ep['wave'], ep['flux']) / np.where(ref_flux != 0, ref_flux, np.nan)
            axes[1].plot(ref_wave, ratio, color=ep['color'], lw=1.2, alpha=0.8,
                         label=f"MJD={ep['mjd']}/MJD={epoch_data[0]['mjd']}")
        axes[1].axhline(1.0, color='gray', ls='--', lw=1.2)
        axes[1].set_ylabel('Flux ratio', fontsize=11, fontweight='bold')
        axes[1].set_ylim(0.0, 3.0)
    else:
        axes[1].set_visible(False)

    if epoch_data:
        add_line_markers_to_plot(axes[0], epoch_data[0]['z'],
                                  x_range=(wave_min, wave_max))

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(mjd_list),
                                                   vmax=max(mjd_list)))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='MJD', shrink=0.6, pad=0.01)

    axes[0].set_ylabel(r'Flux [$10^{-17}$ erg/s/cm$^2$/A]',
                        fontsize=11, fontweight='bold')
    axes[0].set_title(f'Epoch comparison - SDSS_ID {sdss_id} ({n_epochs} observations)',
                       fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2, framealpha=0.9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(wave_min, wave_max)
    axes[1].set_xlabel(r'Wavelength [A]', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(wave_min, wave_max)

    plt.tight_layout()

    if save_dir is not None:
        outdir = Path(save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{sdss_id}_epoch_comparison.pdf"
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {outpath}")

    plt.show()
    return fig


def plot_population(results_list, x_param='z_best', y_param='jet_frac',
                    color_by='best_label', save_dir=None):
    """
    Population-level scatter plot and distribution from a list of cached results.

    Parameters
    ----------
    results_list : list of dicts from cache.load_object_results()
    x_param      : 'z_best', 'pl_alpha', 'pl_delta', 'aicc_margin', 'sn', 'jet_frac'
    y_param      : same options as x_param
    color_by     : 'best_label' or 'fermi_class'
    save_dir     : str or None
    """
    def extract(res, param):
        lm   = res['lmfit_results']
        comp = res['components']
        m    = res['metadata']
        if param == 'z_best':     return lm['z_best']
        elif param == 'jet_frac':
            best = lm['best_label']
            if best == 'Powerlaw+Galaxy' and comp['galpl_contrib']:
                return comp['galpl_contrib']['frac_pl']
            elif best.startswith('Powerlaw+QSO') and comp['qsopl_contrib']:
                return comp['qsopl_contrib']['frac_pl']
            elif best.startswith('Powerlaw+Lines') and comp['linepl_contrib']:
                return comp['linepl_contrib']['frac_pl']
            return np.nan
        elif param == 'pl_alpha':    return lm.get('pl_alpha', np.nan)
        elif param == 'pl_delta':    return lm.get('pl_delta', np.nan)
        elif param == 'aicc_margin': return lm.get('aicc_margin', np.nan)
        elif param == 'sn':          return m['sn_median']
        return np.nan

    def get_color_label(res):
        lm = res['lmfit_results']
        m  = res['metadata']
        if color_by == 'best_label':
            best = lm['best_label']
            if best == 'Powerlaw+Galaxy':            return 'PL+Galaxy',  'steelblue'
            elif best == 'Powerlaw':                 return 'Powerlaw',   'orange'
            elif best.startswith('Powerlaw+QSO'):    return 'PL+QSO',    'purple'
            elif best.startswith('Powerlaw+Lines'):  return 'PL+Lines',  'cyan'
            elif best == 'Galaxy':                   return 'Galaxy',     'green'
            elif best == 'QSO':                      return 'QSO',        'blue'
            else:                                    return 'Other',      'gray'
        elif color_by == 'fermi_class':
            fc = m['fermi_class'].lower()
            if fc == 'bll':    return 'BL Lac', 'steelblue'
            elif fc == 'fsrq': return 'FSRQ',   'tomato'
            elif fc == 'bcu':  return 'BCU',    'goldenrod'
            else:              return fc,        'gray'
        return 'Unknown', 'gray'

    x_vals, y_vals, labels, colors_list = [], [], [], []
    for res in results_list:
        x = extract(res, x_param)
        y = extract(res, y_param)
        if np.isfinite(x) and np.isfinite(y):
            lab, col = get_color_label(res)
            x_vals.append(x); y_vals.append(y)
            labels.append(lab); colors_list.append(col)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    axis_labels = {
        'z_best': 'Redshift z', 'jet_frac': 'Jet fraction',
        'pl_alpha': 'PL slope alpha', 'pl_delta': 'PL curvature delta',
        'aicc_margin': 'AICc margin', 'sn': 'S/N',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    seen = set()
    for x, y, lab, col in zip(x_vals, y_vals, labels, colors_list):
        if lab not in seen:
            ax.scatter(x, y, color=col, s=20, alpha=0.6, label=lab)
            seen.add(lab)
        else:
            ax.scatter(x, y, color=col, s=20, alpha=0.6)

    ax.set_xlabel(axis_labels.get(x_param, x_param), fontsize=12)
    ax.set_ylabel(axis_labels.get(y_param, y_param), fontsize=12)
    ax.set_title(f'{axis_labels.get(y_param, y_param)} vs '
                 f'{axis_labels.get(x_param, x_param)} (N={len(x_vals)})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    unique_labels  = list(dict.fromkeys(labels))
    unique_colors  = {lab: col for lab, col in zip(labels, colors_list)}
    for lab in unique_labels:
        mask = np.array(labels) == lab
        ax2.hist(y_vals[mask], bins=20, alpha=0.5,
                 color=unique_colors[lab], label=lab,
                 edgecolor='black', linewidth=0.5)

    ax2.set_xlabel(axis_labels.get(y_param, y_param), fontsize=12)
    ax2.set_ylabel('N', fontsize=12)
    ax2.set_title(f'Distribution of {axis_labels.get(y_param, y_param)}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_dir is not None:
        outdir = Path(save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"population_{x_param}_vs_{y_param}.pdf"
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    return fig


def plot_ew_summary(results, min_snr=3.0, save_dir=None):
    """
    Scatter plot of detected equivalent widths with S/N labels.

    Emission lines shown in blue, absorption lines in green.
    The 5 A BL Lac/FSRQ boundary is marked with a red dashed line.
    Log scale y-axis for clarity across a wide dynamic range.

    Parameters
    ----------
    results  : dict - output of RedshiftResultsCache.load_object_results()
    min_snr  : float - minimum S/N threshold (default 3.0)
    save_dir : str or None

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd
    meta  = results['metadata']
    spec  = results['spectrum']
    lmfit = results['lmfit_results']
    z     = lmfit['z_best']

    results_ew = compute_EW_for_all_lines(
        spec['common_wave'], spec['flux_resamp'],
        spec['err_resamp'], spec['fit_mask'], z)

    emission_det = [
        (name, abs(ew), snr)
        for name, obs, ew, ew_err, snr, det, ltype, window in results_ew
        if det and snr >= min_snr and np.isfinite(ew) and ltype == 'emission'
    ]
    absorption_det = [
        (name, abs(ew), snr)
        for name, obs, ew, ew_err, snr, det, ltype, window in results_ew
        if det and snr >= min_snr and np.isfinite(ew) and ltype == 'absorption'
    ]

    all_det = emission_det + absorption_det
    if not all_det:
        print(f"No lines detected above S/N={min_snr} for SDSS_ID {meta['SDSS_ID']}")
        return None

    colors = {'Emission': 'steelblue', 'Absorption': 'forestgreen'}

    fig, ax = plt.subplots(figsize=(max(8, len(all_det) * 1.2), 6))

    for i, (name, ew, snr) in enumerate(all_det):
        ltype = 'Emission' if (name, ew, snr) in emission_det else 'Absorption'
        color = colors[ltype]
        pos   = i + 1

        ax.scatter(pos, ew, color=color, s=140, zorder=5,
                   edgecolors='black', linewidths=1.2)
        ax.text(pos, ew * 1.2, f'S/N={snr:.1f}',
                ha='center', va='bottom', fontsize=8,
                color='black', fontweight='bold')

    ax.axhline(5, color='red', ls='--', lw=2, alpha=0.8,
               label='|EW| = 5 A (BL Lac / FSRQ boundary)')
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue',   edgecolor='black', alpha=0.8, label='Emission'),
        Patch(facecolor='forestgreen', edgecolor='black', alpha=0.8, label='Absorption'),
        plt.Line2D([0], [0], color='red', ls='--', lw=2, label='|EW| = 5 A'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.95)

    all_names = [r[0] for r in all_det]
    ax.set_xticks(range(1, len(all_det) + 1))
    ax.set_xticklabels(all_names, fontsize=10, fontweight='bold',
                        rotation=30, ha='right')
    ax.set_ylabel('|Equivalent Width| (A)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Detected spectral lines - SDSS_ID {meta["SDSS_ID"]} (z={z:.4f})\n'
        f'{lmfit["best_label"]} | S/N threshold = {min_snr}',
        fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=10, direction='in', top=True, right=True)
    ax.set_yscale('log')

    all_ews = [r[1] for r in all_det]
    ax.set_ylim(0.5, max(all_ews) * 4)

    plt.tight_layout()

    if save_dir is not None:
        outdir = Path(save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{meta['SDSS_ID']}_ew_summary.pdf"
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    return fig
