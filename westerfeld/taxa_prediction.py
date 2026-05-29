import numpy as np
import pandas as pd

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


def predict_abundance(features, runs, rng):
    """Predict each prevalent taxon's (mCLR) abundance from the other taxa."""
    results = []
    importances = pd.DataFrame(index=features.columns)
    for run, target in enumerate(_sample_targets(features.columns, runs, rng)):
        print(f"{run + 1:>8}  {target}")

        X = features.drop(columns=[target])
        y = features[target].to_numpy()

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        train_trues, train_preds, test_trues, test_preds = [], [], [], []
        fold_importances = np.zeros(X.shape[1])
        for train_index, test_index in kfold.split(X):
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf.fit(X.iloc[train_index], y[train_index])
            train_trues.extend(y[train_index])
            train_preds.extend(rf.predict(X.iloc[train_index]))
            test_trues.extend(y[test_index])
            test_preds.extend(rf.predict(X.iloc[test_index]))
            fold_importances += rf.feature_importances_

        importances[target] = pd.Series(
            fold_importances / kfold.get_n_splits(), index=X.columns
        )
        results.append(
            {
                "taxon": target,
                "r2_train": r2_score(train_trues, train_preds),
                "r2_test": r2_score(test_trues, test_preds),
            }
        )
    return pd.DataFrame(results), importances


def predict_presence(features, presence, targets, runs, rng):
    """Predict each band taxon's presence/absence from the prevalent taxa."""
    results = []
    importances = pd.DataFrame(index=features.columns)
    for run, target in enumerate(_sample_targets(targets, runs, rng)):
        print(f"{run + 1:>8}  {target}")

        X = features.drop(columns=[target], errors="ignore")
        y = presence[target].to_numpy()

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        test_trues, test_scores = [], []
        fold_importances = np.zeros(X.shape[1])
        for train_index, test_index in kfold.split(X):
            clf = RandomForestClassifier(random_state=42, n_jobs=-1)
            clf.fit(X.iloc[train_index], y[train_index])
            test_trues.extend(y[test_index])
            test_scores.extend(positive_class_proba(clf, X.iloc[test_index]))
            fold_importances += clf.feature_importances_

        importances[target] = pd.Series(
            fold_importances / kfold.get_n_splits(), index=X.columns
        )
        results.append(
            {
                "taxon": target,
                "auc_test": auc_or_nan(test_trues, test_scores),
            }
        )
    return pd.DataFrame(results), importances


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

    print("Predicting abundance for every prevalent taxon...")
    regression, regression_importances = predict_abundance(features, runs, rng)
    print("DONE")

    low, high = presence_band
    band_taxa = prevalence[(prevalence >= low) & (prevalence <= high)].index
    print(f"Predicting presence for band taxa ({low}-{high}); {len(band_taxa)} taxa...")
    classification, classification_importances = predict_presence(
        features, presence, band_taxa, runs, rng
    )
    print("DONE")

    return TaxaPredictionResult(
        regression=regression,
        classification=classification,
        regression_importances=regression_importances,
        classification_importances=classification_importances,
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


def dominating_taxa(importances, results, metric, threshold):
    """Rank taxa by mean importance across the models that generalize."""
    passing = results.loc[results[metric] > threshold, "taxon"].tolist()
    subset = importances[passing]
    ranking = pd.DataFrame(
        {
            "mean_importance": subset.mean(axis=1),
            "n_models": subset.notna().sum(axis=1).astype(int),
        }
    )
    ranking.index.name = "taxon"
    return ranking.sort_values("mean_importance", ascending=False)


def main():
    print("-------------------")
    print("| TAXA PREDICTION |")
    print("-------------------")

    result = taxa_prediction(
        "Fungi",
        "Species",
        years=2019,
        habitats="Field_Soil",
        beneficials="Control",
        runs=64,
    )

    regression_summary = summarize_taxa_prediction(result.regression)
    classification_summary = summarize_taxa_prediction(result.classification)
    print(regression_summary)
    print(classification_summary)

    print(predictability_summary(result.regression, "r2_test", 0.5))
    print(predictability_summary(result.classification, "auc_test", 0.7))

    abundance_dominating = dominating_taxa(
        result.regression_importances, result.regression, "r2_test", 0.5
    )
    presence_dominating = dominating_taxa(
        result.classification_importances, result.classification, "auc_test", 0.7
    )
    print(abundance_dominating.head(15))
    print(presence_dominating.head(15))

    result.regression.to_csv("taxa_prediction_abundance.csv", index=False)
    result.classification.to_csv("taxa_prediction_presence.csv", index=False)
    regression_summary.to_csv("taxa_prediction_abundance_summary.csv")
    classification_summary.to_csv("taxa_prediction_presence_summary.csv")
    abundance_dominating.to_csv("taxa_prediction_abundance_importance.csv")
    presence_dominating.to_csv("taxa_prediction_presence_importance.csv")


if __name__ == "__main__":
    main()
