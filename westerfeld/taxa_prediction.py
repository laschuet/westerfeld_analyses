import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from _preparation import relative_abundances


def taxa_prediction(
    type_label,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    runs=None,
):
    _, df, _, _ = relative_abundances(type_label, years, habitats, beneficials, crops)

    print("Learning a model for every taxon as a target...")
    targets = df.columns
    if runs is None:
        runs = len(targets)
    runs = min(runs, len(targets))

    rng = np.random.default_rng(42)
    sampled_targets = rng.choice(targets, size=runs, replace=False)

    print(" " * 7 + "#" + "  Taxon")
    print("-" * 8 + "  " + "-" * 32)
    results = []
    for run, target in enumerate(sampled_targets):
        print(f"{run + 1:>8}  {target}")

        X = df.drop(columns=[target])
        y = df[target].to_numpy()

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        y_train_trues = []
        y_test_trues = []
        y_train_preds = []
        y_test_preds = []
        for train_index, test_index in kfold.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train_true = y[train_index]
            y_test_true = y[test_index]

            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train_true)

            y_train_trues.extend(y_train_true)
            y_test_trues.extend(y_test_true)
            y_train_preds.extend(rf.predict(X_train))
            y_test_preds.extend(rf.predict(X_test))

        results.append(
            {
                "taxon": target,
                "r2_train": r2_score(y_train_trues, y_train_preds),
                "r2_test": r2_score(y_test_trues, y_test_preds),
            }
        )
    print("DONE")

    return pd.DataFrame(results)


def summarize_taxa_prediction(results):
    return pd.DataFrame(
        {
            "mean": [results["r2_train"].mean(), results["r2_test"].mean()],
            "std": [results["r2_train"].std(ddof=0), results["r2_test"].std(ddof=0)],
        },
        index=["r2_train", "r2_test"],
    )


def main():
    print("-------------------")
    print("| TAXA PREDICTION |")
    print("-------------------")

    results = taxa_prediction(
        "Fungi",
        years=2019,
        habitats="Field_Soil",
        beneficials="Control",
        runs=16,
    )

    summary = summarize_taxa_prediction(results)
    print(summary)

    results.to_csv("taxa_prediction.csv", index=False)
    summary.to_csv("taxa_prediction_summary.csv")


if __name__ == "__main__":
    main()
