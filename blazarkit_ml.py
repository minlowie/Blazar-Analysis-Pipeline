"""
blazarkit_ml.py
===============
Machine learning pipeline for Fermi/SDSS-V blazar classification
and redshift reliability prediction.

This is built on top of the blazarkit cache system — by extracting features from
pre-computed spectral fitting results and trains classifiers for:

  1. Blazar classification  (BL Lac / FSRQ / Other)
  2. Redshift reliability   (reliable / uncertain)

Usage
-----
    from blazarkit_ml import BlaZarML
    from blazarkit import NAKBlaZarCache

    cache = NAKBlaZarCache(cache_dir='/path/to/cache')
    ml    = BlaZarML(cache)
    ml.extract_features()
    ml.train()
    ml.predict(new_sdss_id, mjd)

Reference
---------
    Nlowie et al. (in prep.)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') #This is to avoid or reduce not useful warnings

from blazarkit import NAKBlaZarCache, measure_ew


# -- Spectral line list (same as blazarkit) -----------------------------------
EMISSION_LINES = {
    'Ly_alpha': 1215, 'C_IV': 1549, 'C_III': 1909, 'Fe_II': 2600,
    'Mg_II': 2796, 'O_II': 3729, 'H_beta': 4861.3, 'O_III': 5007,
    'H_alpha': 6562.80, 'N_II': 6583.6,
}
ABSORPTION_LINES = {
    'Ca_II_K': 3933.7, 'Ca_II_H': 3968.5, 'Ca_I_G': 4304.40,
    'Mg_b': 5184, 'Na_I_D': 5892.5, 'Ca_Fe': 5269,
}
ALL_LINES = list(EMISSION_LINES.keys()) + list(ABSORPTION_LINES.keys())


# -- Feature extraction --------------------------------------------------------

def extract_features_single(results):
    """
    Extracting the ML features from a single cached results dict.

    Parameters
    ----------
    results : dict — from cache.load()

    Returns
    -------
    features : dict — these are flat feature dictionary; only one level unlike the cache
    """
    meta   = results['metadata']
    lmfit  = results['lmfit_results']
    comp   = results['components']
    spec   = results['spectrum']
    chi2   = results['chi2_grids']
    pz     = results['pz_results']

    feat = {}

    # -- Metadata --------------------------------------------------------------
    feat['SDSS_ID']     = meta['SDSS_ID']
    feat['fermi_class'] = meta['fermi_class']
    feat['obj_class']   = meta['obj_class']
    feat['sn_median']   = float(meta['sn_median'])
    feat['z_sdss']      = float(meta['z_sdss']) if meta['z_sdss'] is not None \
                          and np.isfinite(float(meta['z_sdss'])) else np.nan

    # -- Best-fit model --------------------------------------------------------
    feat['best_label']  = lmfit['best_label']
    feat['z_best']      = float(lmfit['z_best'])
    feat['z_err']       = float(lmfit.get('z_err', np.nan))
    feat['pl_alpha']    = float(lmfit.get('pl_alpha', np.nan))
    feat['pl_delta']    = float(lmfit.get('pl_delta', np.nan))
    feat['aicc_margin'] = float(lmfit.get('aicc_margin', np.nan))
    feat['redchi']      = float(lmfit['best_fit_params']['redchi'])
    feat['chisqr']      = float(lmfit['best_fit_params']['chisqr'])

    # -- Jet fraction ---------------------------------------------------------
    best = lmfit['best_label']
    feat['jet_frac'] = np.nan
    if best == 'Powerlaw+Galaxy' and comp['galpl_contrib'] is not None:
        feat['jet_frac'] = float(comp['galpl_contrib']['frac_pl'])
    elif best.startswith('Powerlaw+QSO') and comp['qsopl_contrib'] is not None:
        feat['jet_frac'] = float(comp['qsopl_contrib']['frac_pl'])
    elif best.startswith('Powerlaw+Lines') and comp['linepl_contrib'] is not None:
        feat['jet_frac'] = float(comp['linepl_contrib']['frac_pl'])
    elif best == 'Powerlaw':
        feat['jet_frac'] = 1.0

    # -- Model family binary flags --------------------------------------------
    feat['is_pl_galaxy'] = int(best == 'Powerlaw+Galaxy')
    feat['is_pl_qso']    = int(best.startswith('Powerlaw+QSO'))
    feat['is_pl_lines']  = int(best.startswith('Powerlaw+Lines'))
    feat['is_pl_only']   = int(best == 'Powerlaw')
    feat['is_galaxy']    = int(best == 'Galaxy')
    feat['is_qso']       = int(best == 'QSO')

    # -- Chi2 ratios (relative model evidence) ---------------------------------
    # Ratio of best chi2 to each family — tells us how much better the
    # best model is than the alternatives
    z_grid   = chi2['z_grid']
    best_chi = float(lmfit['best_fit_params']['redchi'])

    for family, key in [
        ('gal',    'chi2_gal_list'),
        ('qso',    'chi2_qso_list'),
        ('galpl',  'chi2_galpl_list'),
        ('qsopl',  'chi2_qsopl_list'),
        ('linepl', 'chi2_linepl_list'), 
    ]:
        chi_list = chi2.get(key, [])
        if chi_list:
            min_chi = np.nanmin([np.nanmin(c) for c in chi_list
                                 if c is not None and len(c) > 0])
            feat[f'chi2_min_{family}'] = float(min_chi)
            feat[f'chi2_ratio_{family}'] = float(best_chi / min_chi) \
                                           if min_chi > 0 else np.nan
        else:
            feat[f'chi2_min_{family}']   = np.nan
            feat[f'chi2_ratio_{family}'] = np.nan

    pl_chi = chi2.get('chi2_pl')
    if pl_chi is not None:
        min_pl = float(np.nanmin(pl_chi))
        feat['chi2_min_pl']   = min_pl
        feat['chi2_ratio_pl'] = float(best_chi / min_pl) if min_pl > 0 else np.nan
    else:
        feat['chi2_min_pl']   = np.nan
        feat['chi2_ratio_pl'] = np.nan

    # -- p(z) posterior shape features ---------------------------------------
    p_total = np.array(pz['p_total'])
    p_norm  = p_total / np.max(p_total) if np.max(p_total) > 0 else p_total

    feat['z_map']         = float(pz['z_map_total'])
    feat['C_global']      = float(pz.get('C_global', np.nan))

    # Peak height and width of the global posterior
    feat['pz_peak_height'] = float(np.max(p_norm))
    feat['pz_entropy']     = float(-np.nansum(
        p_norm * np.log(p_norm + 1e-10))) if np.any(p_norm > 0) else np.nan

    # Fraction of posterior above 10% of peak — measures how peaked it is
    feat['pz_frac_above_10pct'] = float(np.mean(p_norm > 0.1))

    # Relative contribution of each family to global posterior
    for fam, key in [
        ('gal', 'pz_gal'), ('qso', 'pz_qso'), ('pl', 'pz_pl'),
        ('galpl', 'pz_galpl'), ('qsopl', 'pz_qsopl'), ('linepl', 'pz_linepl'),
    ]:
        p_fam = np.array(pz.get(key, np.zeros_like(p_total)))
        total = np.trapezoid(p_total, z_grid)
        fam_total = np.trapezoid(p_fam, z_grid)
        feat[f'pz_frac_{fam}'] = float(fam_total / total) \
                                  if total > 0 else np.nan

    # -- EW features -----------------------------------------------------------
    results_ew = measure_ew(
        spec['common_wave'], spec['flux_resamp'],
        spec['err_resamp'],  spec['fit_mask'],
        lmfit['z_best']
    )

    n_emission   = 0
    n_absorption = 0

    for name, obs, ew, ew_err, snr, detected, ltype, window in results_ew:
        clean_name = name.replace(' ', '_').replace('[', '').replace(']', '')
        feat[f'ew_{clean_name}']          = float(ew)   if np.isfinite(ew)   else np.nan
        feat[f'ew_err_{clean_name}']      = float(ew_err) if np.isfinite(ew_err) else np.nan
        feat[f'snr_{clean_name}']         = float(snr)
        feat[f'detected_{clean_name}']    = int(detected)
        if detected:
            if ltype == 'emission':
                n_emission += 1
            else:
                n_absorption += 1

    feat['n_emission_detected']   = n_emission
    feat['n_absorption_detected'] = n_absorption
    feat['n_lines_detected']      = n_emission + n_absorption

    # -- Classification target labels -----------------------------------------------
    if best in ('Powerlaw+Galaxy', 'Powerlaw'):
        feat['blazar_class'] = 'BL Lac'
    elif best.startswith('Powerlaw+QSO') or best.startswith('Powerlaw+Lines'):
        feat['blazar_class'] = 'FSRQ'
    else:
        feat['blazar_class'] = 'Other'

    # Redshift reliability label — flag low-confidence redshifts
    z_reliable = (
        np.isfinite(feat['z_best']) and
        feat['z_best'] > 0.01 and
        feat['z_best'] < 5.0 and
        feat['sn_median'] >= 3.0
    )
    feat['z_reliable'] = int(z_reliable)

    return feat


def extract_features_all(cache, verbose=True):
    """
    Extract features from all sources in the local cache.

    Parameters
    ----------
    cache   : NAKBlaZarCache instance
    verbose : bool — printing progress

    Returns
    -------
    df : pd — one row per source, columns are features
    """
    files   = cache.list_cached_objects()
    records = []
    failed  = []

    print(f"Extracting features from {len(files)} cached sources...")

    for i, fname in enumerate(sorted(files)):
        parts   = fname.replace('.pkl.gz', '').split('_')
        sdss_id = parts[1]
        mjd     = int(parts[2]) if len(parts) > 2 else None

        try:
            results = cache.load(sdss_id, mjd=mjd)
            feat    = extract_features_single(results)
            records.append(feat)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(files)}")
        except Exception as e:
            failed.append(fname)
            if verbose:
                print(f"  Failed: {fname} — {e}")

    df = pd.DataFrame(records)

    print(f"\nExtracted features for {len(df)} sources")
    if failed:
        print(f"Failed: {len(failed)} sources")
    print(f"Feature matrix shape: {df.shape}")
    print(f"\nClass breakdown:")
    print(df['blazar_class'].value_counts().to_string())

    return df


#--- the champion ML Pipeline ): -----------------------------------------------------

class BlaZarML:
    """
    The machine learning pipeline for blazar classification and
    redshift reliability prediction.

    Parameters
    ----------
    cache : NAKBlaZarCache instance

    Usage
    -----
        ml = BlaZarML(cache)
        ml.extract_features()       # build feature matrix from cache
        ml.train()                  # train classifier
        ml.evaluate()               # cross-validated performance
        ml.feature_importance()     # which features matter most
        ml.predict('20570296', 60027)  # classify a new source
    """

    # Features used for classification
    CLASSIFICATION_FEATURES = [
        'pl_alpha', 'pl_delta', 'jet_frac', 'aicc_margin', 'redchi',
        'sn_median', 'z_best',
        'chi2_ratio_gal', 'chi2_ratio_qso', 'chi2_ratio_galpl',
        'chi2_ratio_qsopl', 'chi2_ratio_linepl', 'chi2_ratio_pl',
        'pz_frac_gal', 'pz_frac_qso', 'pz_frac_galpl',
        'pz_frac_qsopl', 'pz_frac_linepl',
        'pz_entropy', 'pz_frac_above_10pct',
        'n_emission_detected', 'n_absorption_detected',
        'ew_Mg_II', 'ew_Ca_II_K', 'ew_H_alpha', 'ew_H_beta',
        'snr_Mg_II', 'snr_Ca_II_K', 'snr_H_alpha',
        'detected_Mg_II', 'detected_Ca_II_K', 'detected_H_alpha',
        'detected_H_beta', 'detected_C_IV', 'detected_O_III',
    ]

    def __init__(self, cache):
        self.cache      = cache
        self.df         = None
        self.classifier = None
        self.scaler     = None
        self.label_enc  = None
        self._is_trained = False

    def extract_features(self, save_path=None):
        """Extract features from all cached sources."""
        self.df = extract_features_all(self.cache, verbose=True)
        if save_path:
            self.df.to_csv(save_path, index=False)
            print(f"Features saved to: {save_path}")
        return self.df

    def load_features(self, path):
        """Load previously extracted features from CSV."""
        self.df = pd.read_csv(path)
        print(f"Loaded features: {self.df.shape}")
        return self.df

    def _prepare_X_y(self, target='blazar_class', features=None):
        """Prepare feature matrix X and target vector y."""
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        if features is None:
            features = [f for f in self.CLASSIFICATION_FEATURES
                        if f in self.df.columns]

        # Drop rows with too many NaN features
        df_clean = self.df.dropna(subset=[target])
        X = df_clean[features].copy()

        # Fill remaining NaNs with column median
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        y_raw = df_clean[target].values

        # Encode labels
        self.label_enc = LabelEncoder()
        y = self.label_enc.fit_transform(y_raw)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._feature_names = features
        return X_scaled, y, df_clean

    def train(self, target='blazar_class', model='random_forest'):
        """
        Train the classifier.

        Parameters
        ----------
        target : str — 'blazar_class' or 'z_reliable'
        model  : str — 'random_forest', 'xgboost', or 'gradient_boosting'
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        assert self.df is not None, "Run extract_features() first"

        X, y, df_clean = self._prepare_X_y(target=target)

        print(f"\nTraining {model} classifier for '{target}'...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Classes: {self.label_enc.classes_}")

        if model == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
        elif model == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42
            )
        elif model == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.classifier = XGBClassifier(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, use_label_encoder=False,
                    eval_metric='mlogloss', random_state=42
                )
            except ImportError:
                print("  xgboost not installed — falling back to random_forest")
                self.classifier = RandomForestClassifier(
                    n_estimators=200, random_state=42)

        self.classifier.fit(X, y)
        self._is_trained = True
        self._target     = target
        print(f"  Training complete.")
        return self

    def evaluate(self, cv=5):
        """
        Cross-validated performance evaluation.

        Parameters
        ----------
        cv : int — number of cross-validation folds
        """
        from sklearn.model_selection import cross_validate, StratifiedKFold
        from sklearn.metrics import classification_report

        assert self._is_trained, "Run train() first"

        X, y, _ = self._prepare_X_y(target=self._target)

        cv_results = cross_validate(
            self.classifier, X, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=['accuracy', 'f1_macro', 'f1_weighted'],
            return_train_score=True
        )

        print(f"\nCross-validation results ({cv}-fold):")
        print(f"  Accuracy:     {cv_results['test_accuracy'].mean():.3f} "
              f"± {cv_results['test_accuracy'].std():.3f}")
        print(f"  F1 (macro):   {cv_results['test_f1_macro'].mean():.3f} "
              f"± {cv_results['test_f1_macro'].std():.3f}")
        print(f"  F1 (weighted):{cv_results['test_f1_weighted'].mean():.3f} "
              f"± {cv_results['test_f1_weighted'].std():.3f}")

        # Full classification report on all data
        y_pred = self.classifier.predict(X)
        print(f"\nClassification report (full training set):")
        print(classification_report(
            y, y_pred,
            target_names=self.label_enc.classes_
        ))
        return cv_results

    def feature_importance(self, top_n=20, plot=True):
        """
        Show the most important features for classification.

        Parameters
        ----------
        top_n : int — number of top features to show
        plot  : bool — show bar chart
        """
        import matplotlib.pyplot as plt

        assert self._is_trained, "Run train() first"
        assert hasattr(self.classifier, 'feature_importances_'), \
            "Classifier does not support feature importance"

        importances = self.classifier.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]
        names       = [self._feature_names[i] for i in indices]
        values      = importances[indices]

        print(f"\nTop {top_n} most important features:")
        for i, (name, val) in enumerate(zip(names, values)):
            print(f"  {i+1:>3}. {name:<35} {val:.4f}")

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(top_n), values[::-1],
                    color='steelblue', alpha=0.8, edgecolor='black', lw=0.5)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(names[::-1], fontsize=9)
            ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Features — {self._target}',
                         fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            plt.show()

        return dict(zip(names, values))

    def predict(self, sdss_id, mjd=None, verbose=True):
        """
        Predict the blazar class for a source from the cache.

        Parameters
        ----------
        sdss_id : str
        mjd     : int or None
        verbose : bool — print result

        Returns
        -------
        prediction : dict — class, probabilities, features used
        """
        assert self._is_trained, "Run train() first"

        results = self.cache.load(sdss_id, mjd=mjd)
        feat    = extract_features_single(results)

        X = np.array([[feat.get(f, np.nan) for f in self._feature_names]])
        # Fill NaNs with training median
        for j, fname in enumerate(self._feature_names):
            if np.isnan(X[0, j]):
                col_vals = self.df[fname].dropna()
                X[0, j]  = col_vals.median() if len(col_vals) > 0 else 0.0

        X_scaled = self.scaler.transform(X)
        pred_idx  = self.classifier.predict(X_scaled)[0]
        pred_class = self.label_enc.inverse_transform([pred_idx])[0]

        probs = {}
        if hasattr(self.classifier, 'predict_proba'):
            prob_arr = self.classifier.predict_proba(X_scaled)[0]
            probs    = {cls: float(p) for cls, p in
                        zip(self.label_enc.classes_, prob_arr)}

        if verbose:
            print(f"\n{'='*50}")
            print(f"SDSS_ID: {sdss_id}  MJD: {mjd}")
            print(f"Pipeline label: {feat['best_label']}")
            print(f"ML prediction:  {pred_class}")
            if probs:
                print(f"Probabilities:")
                for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
                    bar = '█' * int(p * 20)
                    print(f"  {cls:<10} {p:.3f}  {bar}")
            print(f"{'='*50}")

        return {
            'SDSS_ID'        : sdss_id,
            'MJD'            : mjd,
            'pipeline_label' : feat['best_label'],
            'ml_prediction'  : pred_class,
            'probabilities'  : probs,
            'features'       : feat,
        }

    def predict_all(self, save_path=None):
        """
        Run predictions on all sources in the feature matrix.

        Returns
        -------
        df_pred : DataFrame with ML predictions added
        """
        assert self._is_trained, "Run train() first"
        assert self.df is not None, "Run extract_features() first"

        features = [f for f in self._feature_names if f in self.df.columns]
        X = self.df[features].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        X_scaled   = self.scaler.transform(X)
        pred_idx   = self.classifier.predict(X_scaled)
        pred_class = self.label_enc.inverse_transform(pred_idx)

        df_pred = self.df.copy()
        df_pred['ml_prediction'] = pred_class

        if hasattr(self.classifier, 'predict_proba'):
            prob_arr = self.classifier.predict_proba(X_scaled)
            for i, cls in enumerate(self.label_enc.classes_):
                df_pred[f'prob_{cls}'] = prob_arr[:, i]

        df_pred['pipeline_agrees'] = (
            df_pred['blazar_class'] == df_pred['ml_prediction']
        ).astype(int)

        n_agree = df_pred['pipeline_agrees'].sum()
        print(f"\nPredictions on {len(df_pred)} sources:")
        print(f"  Pipeline agrees with ML: {n_agree}/{len(df_pred)} "
              f"({100*n_agree/len(df_pred):.1f}%)")
        print(f"\nML prediction breakdown:")
        print(df_pred['ml_prediction'].value_counts().to_string())

        if save_path:
            df_pred.to_csv(save_path, index=False)
            print(f"\nSaved to: {save_path}")

        return df_pred

    def confusion_matrix(self):
        """Plot confusion matrix for training set predictions."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        assert self._is_trained, "Run train() first"

        X, y, _ = self._prepare_X_y(target=self._target)
        y_pred  = self.classifier.predict(X)

        cm  = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.label_enc.classes_
        )
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'Confusion Matrix — {self._target}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return cm

    def save(self, path='blazarkit_ml_model.pkl'):
        """Save the trained model to disk."""
        import pickle
        assert self._is_trained, "Run train() first"
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier'    : self.classifier,
                'scaler'        : self.scaler,
                'label_enc'     : self.label_enc,
                'feature_names' : self._feature_names,
                'target'        : self._target,
            }, f)
        print(f"Model saved to: {path}")

    def load_model(self, path='blazarkit_ml_model.pkl'):
        """Load a previously trained model from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.classifier      = data['classifier']
        self.scaler          = data['scaler']
        self.label_enc       = data['label_enc']
        self._feature_names  = data['feature_names']
        self._target         = data['target']
        self._is_trained     = True
        print(f"Model loaded from: {path}")
        return self


# ── Variability and Changing-Look Blazar Analysis ─────────────────────────────

def extract_variability_features(sdss_id, mjd_list, cache, verbose=True):
    """
    Extract variability features for a multiply-observed source by comparing
    spectral fitting parameters across epochs.

    For each source with multiple observations we compute statistics
    (mean, std, range) of key parameters across epochs. Changes in model
    family or blazar class between epochs flag potential changing-look
    blazar candidates.

    Parameters
    ----------
    sdss_id  : str  - SDSS_ID of the source
    mjd_list : list - MJD values for each epoch
    cache    : NAKBlaZarCache instance
    verbose  : bool - print progress

    Returns
    -------
    var_feat : dict - variability feature dictionary for this source
    """
    epoch_feats = []
    failed_mjds = []

    for mjd in sorted(mjd_list):
        try:
            results = cache.load(str(sdss_id), mjd=mjd)
            feat    = extract_features_single(results)
            feat['mjd'] = mjd
            epoch_feats.append(feat)
        except Exception as e:
            failed_mjds.append(mjd)
            if verbose:
                print(f"  Failed MJD={mjd}: {e}")

    if len(epoch_feats) < 2:
        if verbose:
            print(f"  {sdss_id}: fewer than 2 epochs loaded — skipping")
        return None

    # ── Basic info ────────────────────────────────────────────────────────────
    var_feat = {
        'SDSS_ID'         : str(sdss_id),
        'n_epochs'        : len(epoch_feats),
        'baseline_days'   : max(e['mjd'] for e in epoch_feats) -
                            min(e['mjd'] for e in epoch_feats),
        'fermi_class'     : epoch_feats[0]['fermi_class'],
        'sn_median_mean'  : np.nanmean([e['sn_median'] for e in epoch_feats]),
    }

    # ── Parameter variability statistics ─────────────────────────────────────
    # For each key spectral parameter, compute statistics across epochs
    for param in ['pl_alpha', 'pl_delta', 'jet_frac', 'z_best',
                  'redchi', 'aicc_margin', 'n_emission_detected',
                  'n_absorption_detected', 'ew_Mg_II', 'snr_Mg_II',
                  'chi2_ratio_galpl', 'chi2_ratio_qsopl', 'pz_entropy']:
        vals = [e.get(param, np.nan) for e in epoch_feats]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2:
            var_feat[f'{param}_mean']  = float(np.mean(vals))
            var_feat[f'{param}_std']   = float(np.std(vals))
            var_feat[f'{param}_range'] = float(np.max(vals) - np.min(vals))
            var_feat[f'{param}_min']   = float(np.min(vals))
            var_feat[f'{param}_max']   = float(np.max(vals))
        else:
            for suffix in ['mean', 'std', 'range', 'min', 'max']:
                var_feat[f'{param}_{suffix}'] = np.nan

    # ── Model family changes ──────────────────────────────────────────────────
    # Track whether the best-fit model or blazar class changed between epochs
    labels  = [e['best_label']   for e in epoch_feats]
    classes = [e['blazar_class'] for e in epoch_feats]

    var_feat['label_changed']      = int(len(set(labels))  > 1)
    var_feat['class_changed']      = int(len(set(classes)) > 1)
    var_feat['n_unique_labels']    = len(set(labels))
    var_feat['n_unique_classes']   = len(set(classes))
    var_feat['most_common_class']  = max(set(classes), key=classes.count)

    # First and last epoch model labels — captures direction of change
    var_feat['first_epoch_label']  = labels[0]
    var_feat['last_epoch_label']   = labels[-1]
    var_feat['first_epoch_class']  = classes[0]
    var_feat['last_epoch_class']   = classes[-1]

    # ── Jet fraction variability ──────────────────────────────────────────────
    # Special handling — NaN jet_frac means Galaxy/QSO model (no jet)
    # We treat this as jet_frac=0 for variability purposes
    jet_vals = []
    for e in epoch_feats:
        jf = e.get('jet_frac', np.nan)
        if np.isfinite(jf):
            jet_vals.append(jf)
        elif e['best_label'] in ('Galaxy', 'QSO'):
            jet_vals.append(0.0)  # no jet contribution

    if len(jet_vals) >= 2:
        var_feat['jet_frac_variability'] = float(np.std(jet_vals))
        var_feat['jet_frac_trend']       = float(jet_vals[-1] - jet_vals[0])
    else:
        var_feat['jet_frac_variability'] = np.nan
        var_feat['jet_frac_trend']       = np.nan

    # ── Changing-look target labels ───────────────────────────────────────────
    # A changing-look blazar is one where the blazar class changes between epochs
    # This is the primary scientific target for the variability classifier
    var_feat['is_changing_look'] = int(var_feat['class_changed'] == 1)

    # A variable source is one where key parameters change significantly
    # We use pl_alpha_std > 0.3 as a simple threshold (can be tuned)
    pl_alpha_std = var_feat.get('pl_alpha_std', np.nan)
    jet_frac_std = var_feat.get('jet_frac_std', np.nan)
    var_feat['is_variable'] = int(
        (np.isfinite(pl_alpha_std)  and pl_alpha_std  > 0.3) or
        (np.isfinite(jet_frac_std)  and jet_frac_std  > 0.1)
    )

    if verbose:
        print(f"  {sdss_id}: {len(epoch_feats)} epochs, "
              f"baseline={var_feat['baseline_days']} days, "
              f"class_changed={var_feat['class_changed']}, "
              f"is_variable={var_feat['is_variable']}")

    return var_feat


def extract_variability_features_all(multi_match_table, cache, verbose=True):
    """
    Extract variability features for all multiply-observed sources.

    Parameters
    ----------
    multi_match_table : astropy Table — the multiple-match table with
                        columns SDSS_ID and MJD
    cache             : NAKBlaZarCache instance
    verbose           : bool

    Returns
    -------
    df_var : pandas DataFrame — one row per unique source
    """
    from collections import defaultdict

    # Group MJDs by SDSS_ID
    source_mjds = defaultdict(list)
    for row in multi_match_table:
        source_mjds[str(row['SDSS_ID'])].append(int(row['MJD']))

    print(f"Found {len(source_mjds)} unique sources in multi-match table")
    print(f"Extracting variability features...")

    records = []
    for i, (sdss_id, mjd_list) in enumerate(source_mjds.items()):
        if len(mjd_list) < 2:
            continue  # need at least 2 epochs
        var_feat = extract_variability_features(
            sdss_id, mjd_list, cache, verbose=verbose)
        if var_feat is not None:
            records.append(var_feat)

    df_var = pd.DataFrame(records)

    print(f"\nExtracted variability features for {len(df_var)} sources")
    print(f"Feature matrix shape: {df_var.shape}")
    print(f"\nChanging-look candidates: {df_var['is_changing_look'].sum()}")
    print(f"Variable sources:          {df_var['is_variable'].sum()}")
    print(f"Stable sources:            {(df_var['is_variable'] == 0).sum()}")

    return df_var


class BlaZarVariabilityML:
    """
    Machine learning pipeline for blazar variability prediction.

    Trained on multiply-observed sources from the multiple-match cache,
    this classifier predicts:
      1. Whether a source is variable (is_variable)
      2. Whether a source is a changing-look blazar candidate (is_changing_look)

    Parameters
    ----------
    cache : NAKBlaZarCache instance pointing to the multiple-match cache

    Usage
    -----
        from blazarkit_ml import BlaZarVariabilityML
        from astropy.table import Table
        from astropy.io import fits

        multi_cache = NAKBlaZarCache(cache_dir='/path/to/multi_match_cache')
        multi_table = Table(fits.open('Sources_with_multiple_matches.fits')[1].data)

        vml = BlaZarVariabilityML(multi_cache)
        vml.extract_features(multi_table)
        vml.train(target='is_variable')
        vml.evaluate()
        vml.feature_importance()
    """

    # Variability features used for classification
    VARIABILITY_FEATURES = [
        'n_epochs', 'baseline_days', 'sn_median_mean',
        'pl_alpha_mean',  'pl_alpha_std',  'pl_alpha_range',
        'pl_delta_mean',  'pl_delta_std',  'pl_delta_range',
        'jet_frac_mean',  'jet_frac_std',  'jet_frac_range',
        'jet_frac_variability', 'jet_frac_trend',
        'redchi_mean',    'redchi_std',    'redchi_range',
        'aicc_margin_mean', 'aicc_margin_std',
        'z_best_mean',    'z_best_std',
        'pz_entropy_mean', 'pz_entropy_std',
        'chi2_ratio_galpl_mean', 'chi2_ratio_galpl_std',
        'chi2_ratio_qsopl_mean', 'chi2_ratio_qsopl_std',
        'ew_Mg_II_mean',  'ew_Mg_II_std',
        'snr_Mg_II_mean', 'snr_Mg_II_std',
        'n_emission_detected_mean', 'n_emission_detected_std',
        'n_unique_labels', 'label_changed',
    ]

    def __init__(self, cache):
        self.cache       = cache
        self.df          = None
        self.classifier  = None
        self.scaler      = None
        self.label_enc   = None
        self._is_trained = False

    def extract_features(self, multi_match_table, save_path=None):
        """Extract variability features from the multiple-match table."""
        self.df = extract_variability_features_all(
            multi_match_table, self.cache, verbose=True)
        if save_path:
            self.df.to_csv(save_path, index=False)
            print(f"Saved to: {save_path}")
        return self.df

    def load_features(self, path):
        """Load previously extracted variability features."""
        self.df = pd.read_csv(path)
        print(f"Loaded variability features: {self.df.shape}")
        return self.df

    def _prepare_X_y(self, target='is_variable'):
        """Prepare feature matrix and target vector."""
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        features  = [f for f in self.VARIABILITY_FEATURES
                     if f in self.df.columns]
        df_clean  = self.df.dropna(subset=[target])
        X         = df_clean[features].copy()

        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        y_raw = df_clean[target].values.astype(int)

        self.label_enc = LabelEncoder()
        y = self.label_enc.fit_transform(y_raw)

        from sklearn.preprocessing import StandardScaler
        self.scaler   = StandardScaler()
        X_scaled      = self.scaler.fit_transform(X)

        self._feature_names = features
        return X_scaled, y, df_clean

    def train(self, target='is_variable', model='random_forest', n_estimators=50):
        """
        Train the variability classifier.

        Parameters
        ----------
        target       : str  - 'is_variable' or 'is_changing_look'
        model        : str  - 'random_forest' or 'gradient_boosting'
        n_estimators : int  - number of trees (default 50)
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        assert self.df is not None, "Run extract_features() first"

        X, y, df_clean = self._prepare_X_y(target=target)

        print(f"\nTraining {model} variability classifier for '{target}'...")
        print(f"  Samples:  {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes:  {dict(zip(*np.unique(y, return_counts=True)))}")

        if model == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators  = n_estimators,
                class_weight  = 'balanced',
                random_state  = 42,
                n_jobs        = -1
            )
        elif model == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators  = n_estimators,
                max_depth     = 3,
                learning_rate = 0.1,
                random_state  = 42
            )

        self.classifier.fit(X, y)
        self._is_trained = True
        self._target     = target
        print(f"  Training complete.")
        return self

    def evaluate(self, cv=5):
        """Cross-validated performance evaluation."""
        from sklearn.model_selection import cross_validate, StratifiedKFold
        from sklearn.metrics import classification_report

        assert self._is_trained, "Run train() first"

        X, y, _ = self._prepare_X_y(target=self._target)

        # Use fewer folds if sample is small
        n_folds = min(cv, np.bincount(y).min())
        if n_folds < 2:
            print("  Too few samples in one class for cross-validation.")
            return None

        cv_results = cross_validate(
            self.classifier, X, y,
            cv      = StratifiedKFold(n_splits=n_folds,
                                      shuffle=True, random_state=42),
            scoring = ['accuracy', 'f1_macro', 'f1_weighted'],
            return_train_score = True
        )

        print(f"\nCross-validation results ({n_folds}-fold):")
        print(f"  Accuracy:      {cv_results['test_accuracy'].mean():.3f} "
              f"± {cv_results['test_accuracy'].std():.3f}")
        print(f"  F1 (macro):    {cv_results['test_f1_macro'].mean():.3f} "
              f"± {cv_results['test_f1_macro'].std():.3f}")
        print(f"  F1 (weighted): {cv_results['test_f1_weighted'].mean():.3f} "
              f"± {cv_results['test_f1_weighted'].std():.3f}")

        y_pred = self.classifier.predict(X)
        print(f"\nClassification report:")
        print(classification_report(
            y, y_pred,
            target_names=[str(c) for c in self.label_enc.classes_]
        ))
        return cv_results

    def feature_importance(self, top_n=15, plot=True):
        """Show the most important variability features."""
        import matplotlib.pyplot as plt

        assert self._is_trained, "Run train() first"

        importances = self.classifier.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]
        names       = [self._feature_names[i] for i in indices]
        values      = importances[indices]

        print(f"\nTop {top_n} variability features:")
        for i, (name, val) in enumerate(zip(names, values)):
            print(f"  {i+1:>3}. {name:<40} {val:.4f}")

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(top_n), values[::-1],
                    color='mediumpurple', alpha=0.8,
                    edgecolor='black', lw=0.5)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(names[::-1], fontsize=9)
            ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Variability Features — {self._target}',
                         fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            plt.show()

        return dict(zip(names, values))

    def predict_source(self, sdss_id, mjd_list, verbose=True):
        """
        Predict variability class for a multiply-observed source.

        Parameters
        ----------
        sdss_id  : str  - SDSS_ID
        mjd_list : list - list of MJD values
        verbose  : bool

        Returns
        -------
        prediction : dict
        """
        assert self._is_trained, "Run train() first"

        var_feat = extract_variability_features(
            sdss_id, mjd_list, self.cache, verbose=False)

        if var_feat is None:
            print(f"Not enough epochs for {sdss_id}")
            return None

        X = np.array([[var_feat.get(f, np.nan)
                       for f in self._feature_names]])
        for j, fname in enumerate(self._feature_names):
            if np.isnan(X[0, j]):
                col_vals = self.df[fname].dropna()
                X[0, j]  = col_vals.median() if len(col_vals) > 0 else 0.0

        X_scaled   = self.scaler.transform(X)
        pred_idx   = self.classifier.predict(X_scaled)[0]
        pred_class = self.label_enc.inverse_transform([pred_idx])[0]

        probs = {}
        if hasattr(self.classifier, 'predict_proba'):
            prob_arr = self.classifier.predict_proba(X_scaled)[0]
            probs    = {str(cls): float(p) for cls, p in
                        zip(self.label_enc.classes_, prob_arr)}

        if verbose:
            print(f"\n{'='*55}")
            print(f"SDSS_ID:    {sdss_id}")
            print(f"N epochs:   {var_feat['n_epochs']}")
            print(f"Baseline:   {var_feat['baseline_days']} days")
            print(f"Prediction: {self._target} = {pred_class}")
            if probs:
                for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
                    bar = '█' * int(p * 20)
                    print(f"  {cls}  {p:.3f}  {bar}")
            print(f"{'='*55}")

        return {
            'SDSS_ID'        : sdss_id,
            'n_epochs'       : var_feat['n_epochs'],
            'baseline_days'  : var_feat['baseline_days'],
            'prediction'     : pred_class,
            'probabilities'  : probs,
            'features'       : var_feat,
        }

    def predict_all(self, save_path=None):
        """Run predictions on all sources in the variability feature matrix."""
        assert self._is_trained, "Run train() first"

        features  = [f for f in self._feature_names if f in self.df.columns]
        X         = self.df[features].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        X_scaled   = self.scaler.transform(X)
        pred_idx   = self.classifier.predict(X_scaled)
        pred_class = self.label_enc.inverse_transform(pred_idx)

        df_pred = self.df.copy()
        df_pred['ml_prediction'] = pred_class

        if hasattr(self.classifier, 'predict_proba'):
            prob_arr = self.classifier.predict_proba(X_scaled)
            for i, cls in enumerate(self.label_enc.classes_):
                df_pred[f'prob_{cls}'] = prob_arr[:, i]

        print(f"\nVariability predictions on {len(df_pred)} sources:")
        print(df_pred['ml_prediction'].value_counts().to_string())

        # Show top changing-look candidates
        if 'prob_1' in df_pred.columns:
            print(f"\nTop changing-look candidates by probability:")
            top = df_pred.nlargest(10, 'prob_1')[
                ['SDSS_ID', 'n_epochs', 'baseline_days',
                 'ml_prediction', 'prob_1',
                 'jet_frac_std', 'pl_alpha_std', 'class_changed']]
            print(top.to_string(index=False))

        if save_path:
            df_pred.to_csv(save_path, index=False)
            print(f"\nSaved to: {save_path}")

        return df_pred

    def save(self, path='blazarkit_variability_model.pkl'):
        """Save the trained variability model."""
        import pickle
        assert self._is_trained, "Run train() first"
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier'    : self.classifier,
                'scaler'        : self.scaler,
                'label_enc'     : self.label_enc,
                'feature_names' : self._feature_names,
                'target'        : self._target,
            }, f)
        print(f"Model saved to: {path}")

    def load_model(self, path='blazarkit_variability_model.pkl'):
        """Load a previously trained variability model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.classifier     = data['classifier']
        self.scaler         = data['scaler']
        self.label_enc      = data['label_enc']
        self._feature_names = data['feature_names']
        self._target        = data['target']
        self._is_trained    = True
        print(f"Model loaded from: {path}")
        return self
