import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

from _preparation import (
    common_preparation,
    filter_prevalence,
    mclr,
    rarefied_taxa_table,
    relative_abundances,
)


@dataclass
class TaxaPredictionResult:
    regression: pd.DataFrame
    classification: pd.DataFrame
    regression_importances: pd.DataFrame
    classification_importances: pd.DataFrame
    regression_directions: pd.DataFrame
    classification_directions: pd.DataFrame
    # Per-target out-of-fold SHAP matrices (taxon -> samples x predictors),
    # plus the aligned feature matrix, kept for SHAP summary plots.
    regression_shap: dict
    classification_shap: dict
    features: pd.DataFrame


def positive_class_proba(classifier, X):
    """Predicted probability of presence (class 1), robust to single-class folds."""
    classes = list(classifier.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return classifier.predict_proba(X)[:, classes.index(1)]


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN when only one class is present (AUC is then undefined)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _sample_targets(targets, runs, rng):
    if runs is None:
        runs = len(targets)
    runs = min(runs, len(targets))
    return rng.choice(targets, size=runs, replace=False)


def fold_shap_values(model, X_fold, positive_class=False):
    """
    Per-sample, per-feature SHAP contributions for one held-out fold.

    Returns a (n_samples, n_features) array. For classifiers, TreeSHAP returns
    one set of values per class; we keep the positive class (presence) so the
    sign means "pushes towards presence".
    """
    values = shap.TreeExplainer(model).shap_values(X_fold, check_additivity=False)
    values = np.asarray(values)
    if positive_class and values.ndim == 3:
        # (n_samples, n_features, n_classes) -> positive class
        values = values[:, :, 1]
    return values


def aggregate_shap(df, feature_values):
    """
    Collapse out-of-fold SHAP contributions into per-feature importance.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows = samples (out-of-fold), columns = features, values = SHAP.
    feature_values : pandas.DataFrame
        The feature matrix aligned to `shap_df` (same rows and columns), used
        to recover the direction of each feature's effect.

    Returns
    -------
    importance : pandas.Series
        Mean absolute SHAP value per feature (magnitude of influence).
    direction : pandas.Series
        Sign of the correlation between each feature's value and its SHAP
        contribution: +1 = higher feature value pushes the prediction up
        (co-occurrence), -1 = pushes it down (co-exclusion), 0 = no signal.
    """
    importance = df.abs().mean(axis=0)

    aligned = feature_values.loc[df.index, df.columns]
    direction = {}
    for feature in df.columns:
        contribution = df[feature].to_numpy()
        value = aligned[feature].to_numpy()
        if np.std(contribution) == 0 or np.std(value) == 0:
            direction[feature] = 0.0
        else:
            direction[feature] = float(np.sign(np.corrcoef(value, contribution)[0, 1]))
    return importance, pd.Series(direction)


def predict_abundance(features, runs, rng):
    """Predict each prevalent taxon's (mCLR) abundance from the other taxa."""
    results = []
    importances = pd.DataFrame(index=features.columns)
    directions = pd.DataFrame(index=features.columns)
    shap_frames = {}
    for run, target in enumerate(_sample_targets(features.columns, runs, rng)):
        print(f"{run + 1:>8}  {target}")

        X = features.drop(columns=[target])
        y = features[target].to_numpy()

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        train_trues, train_preds, test_trues, test_preds = [], [], [], []
        fold_shap = []
        for train_index, test_index in kfold.split(X):
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf.fit(X.iloc[train_index], y[train_index])
            train_trues.extend(y[train_index])
            train_preds.extend(rf.predict(X.iloc[train_index]))
            test_trues.extend(y[test_index])
            test_preds.extend(rf.predict(X.iloc[test_index]))
            X_test = X.iloc[test_index]
            fold_shap.append(
                pd.DataFrame(fold_shap_values(rf, X_test), index=X_test.index, columns=X.columns)
            )

        shap_df = pd.concat(fold_shap)
        importance, direction = aggregate_shap(shap_df, X)
        importances[target] = importance
        directions[target] = direction
        shap_frames[target] = shap_df
        results.append(
            {
                "taxon": target,
                "r2_train": r2_score(train_trues, train_preds),
                "r2_test": r2_score(test_trues, test_preds),
            }
        )
    return pd.DataFrame(results), importances, directions, shap_frames


def predict_presence(features, presence, targets, runs, rng):
    """Predict each band taxon's presence/absence from the prevalent taxa."""
    results = []
    importances = pd.DataFrame(index=features.columns)
    directions = pd.DataFrame(index=features.columns)
    shap_frames = {}
    for run, target in enumerate(_sample_targets(targets, runs, rng)):
        print(f"{run + 1:>8}  {target}")

        X = features.drop(columns=[target], errors="ignore")
        y = presence[target].to_numpy()

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        test_trues, test_scores = [], []
        fold_shap = []
        for train_index, test_index in kfold.split(X):
            clf = RandomForestClassifier(random_state=42, n_jobs=-1)
            clf.fit(X.iloc[train_index], y[train_index])
            test_trues.extend(y[test_index])
            test_scores.extend(positive_class_proba(clf, X.iloc[test_index]))
            # A single-class training fold collapses TreeSHAP to one class, so skip it
            if len(np.unique(y[train_index])) < 2:
                continue
            X_test = X.iloc[test_index]
            fold_shap.append(
                pd.DataFrame(
                    fold_shap_values(clf, X_test, positive_class=True),
                    index=X_test.index,
                    columns=X.columns,
                )
            )

        if fold_shap:
            shap_df = pd.concat(fold_shap)
            importance, direction = aggregate_shap(shap_df, X)
        else:
            shap_df = None
            importance = pd.Series(np.nan, index=X.columns)
            direction = pd.Series(np.nan, index=X.columns)
        importances[target] = importance
        directions[target] = direction
        shap_frames[target] = shap_df
        results.append(
            {
                "taxon": target,
                "auc_test": auc_or_nan(test_trues, test_scores),
            }
        )
    return pd.DataFrame(results), importances, directions, shap_frames


def taxa_prediction(
    type_label,
    taxonomy,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    runs=None,
    min_prevalence=0.7,
    presence_band=(0.25, 0.75),
):
    df_long = common_preparation(type_label, years, habitats, beneficials, crops)
    df_abs = rarefied_taxa_table(df_long, taxonomy)
    df = relative_abundances(df_abs)

    prevalence = (df > 0).mean(axis=0)
    presence = (df > 0).astype(int)
    features = mclr(filter_prevalence(df, min_prevalence))
    print(f"Samples: {features.shape[0]}, feature taxa: {features.shape[1]}")

    rng = np.random.default_rng(42)

    print("Predicting relative abundance for every prevalent taxon...")
    (
        regression,
        regression_importances,
        regression_directions,
        regression_shap,
    ) = predict_abundance(features, runs, rng)
    print("DONE")

    low, high = presence_band
    band_taxa = prevalence[(prevalence >= low) & (prevalence <= high)].index
    print(f"Predicting presence for band taxa ({low}-{high}); {len(band_taxa)} taxa...")
    (
        classification,
        classification_importances,
        classification_directions,
        classification_shap,
    ) = predict_presence(features, presence, band_taxa, runs, rng)
    print("DONE")

    return TaxaPredictionResult(
        regression=regression,
        classification=classification,
        regression_importances=regression_importances,
        classification_importances=classification_importances,
        regression_directions=regression_directions,
        classification_directions=classification_directions,
        regression_shap=regression_shap,
        classification_shap=classification_shap,
        features=features,
    )


def summarize_taxa_prediction(results):
    metrics = [column for column in results.columns if column != "taxon"]
    return pd.DataFrame(
        {
            "mean": [results[metric].mean() for metric in metrics],
            "std": [results[metric].std(ddof=0) for metric in metrics],
        },
        index=metrics,
    )


def predictability_summary(results, metric, threshold):
    values = results[metric]
    n = int(values.notna().sum())
    n_above = int((values > threshold).sum())
    return pd.Series(
        {
            "n": n,
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "max": values.max(),
            "threshold": threshold,
            "n_above": n_above,
            "frac_above": n_above / n if n else float("nan"),
        }
    )


def dominating_taxa(importances, results, metric, threshold, directions=None):
    """
    Rank taxa by mean SHAP importance across the models that generalize.

    `mean_importance` is the mean absolute SHAP value (magnitude of influence)
    a taxon exerts as a predictor. When `directions` is given, `mean_direction`
    averages the per-model effect signs (+1 co-occurrence, -1 co-exclusion);
    values near +1/-1 mean a consistent effect, values near 0 mean mixed.
    """
    passing = results.loc[results[metric] > threshold, "taxon"].tolist()
    subset = importances[passing]
    ranking = pd.DataFrame(
        {
            "mean_importance": subset.mean(axis=1),
            "n_models": subset.notna().sum(axis=1).astype(int),
        }
    )
    if directions is not None:
        ranking["mean_direction"] = directions[passing].mean(axis=1)
    ranking.index.name = "taxon"
    return ranking.sort_values("mean_importance", ascending=False)


def plot_shap_summary(shap_df, features, target, ax=None, max_display=15):
    """
    SHAP beeswarm for one target taxon: each predictor's out-of-fold SHAP
    contributions, coloured by the predictor's (mCLR) value. Points right of
    zero pushed the prediction up, left pushed it down.
    """
    feature_values = features.loc[shap_df.index, shap_df.columns]
    explanation = shap.Explanation(
        values=shap_df.to_numpy(),
        data=feature_values.to_numpy(),
        feature_names=list(shap_df.columns),
    )
    if ax is not None:
        plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.gca().set_title(target)


def plot_top_shap_summaries(
    shap_frames, features, results, metric, threshold, path, max_display=15, top_n=6
):
    """
    Grid of SHAP beeswarms for the best-predicted targets (highest `metric`
    among those above `threshold`). Targets without SHAP values are skipped.
    """
    ranked = results[results[metric] > threshold].sort_values(
        metric, ascending=False
    )
    targets = [
        row.taxon
        for row in ranked.itertuples()
        if shap_frames.get(row.taxon) is not None
    ][:top_n]
    if not targets:
        print(f"No targets above {metric} > {threshold}; skipping SHAP summaries.")
        return None

    ncols = min(2, len(targets))
    nrows = math.ceil(len(targets) / ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False
    )
    axs = axs.reshape(-1)
    for ax, target in zip(axs, targets):
        plot_shap_summary(shap_frames[target], features, target, ax=ax, max_display=max_display)
    for ax in axs[len(targets):]:
        ax.set_visible(False)

    fig.suptitle(f"SHAP summary of best-predicted targets ({metric})")
    fig.tight_layout()
    fig.savefig(path)
    return fig


def main():
    print("-------------------")
    print("| TAXA PREDICTION |")
    print("-------------------")

    result = taxa_prediction(
        "Fungi",
        "Genus",
        years=2019,
        habitats="Field_Soil",
        runs=64,
    )

    regression_summary = summarize_taxa_prediction(result.regression)
    classification_summary = summarize_taxa_prediction(result.classification)
    print(regression_summary)
    print(classification_summary)

    print(predictability_summary(result.regression, "r2_test", 0.5))
    print(predictability_summary(result.classification, "auc_test", 0.7))

    abundance_dominating = dominating_taxa(
        result.regression_importances,
        result.regression,
        "r2_test",
        0.5,
        directions=result.regression_directions,
    )
    presence_dominating = dominating_taxa(
        result.classification_importances,
        result.classification,
        "auc_test",
        0.7,
        directions=result.classification_directions,
    )
    print(abundance_dominating.head(15))
    print(presence_dominating.head(15))

    result.regression.to_csv("taxa_prediction_abundance.csv", index=False)
    result.classification.to_csv("taxa_prediction_presence.csv", index=False)
    regression_summary.to_csv("taxa_prediction_abundance_summary.csv")
    classification_summary.to_csv("taxa_prediction_presence_summary.csv")
    abundance_dominating.to_csv("taxa_prediction_abundance_importance.csv")
    presence_dominating.to_csv("taxa_prediction_presence_importance.csv")

    plot_top_shap_summaries(
        result.regression_shap,
        result.features,
        result.regression,
        "r2_test",
        0.5,
        path="taxa_prediction_abundance_shap.pdf",
    )
    plot_top_shap_summaries(
        result.classification_shap,
        result.features,
        result.classification,
        "auc_test",
        0.7,
        path="taxa_prediction_presence_shap.pdf",
    )


if __name__ == "__main__":
    main()
