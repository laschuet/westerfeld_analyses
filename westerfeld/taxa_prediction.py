import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from _preparation import relative_abundances


def taxa_prediction(
    type_label,
    file_name,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    runs=None,
):
    _, df, _, _ = relative_abundances(type_label, years, habitats, beneficials, crops)

    print("Learning models for every species as a target...")
    results = []
    targets = df.columns
    if runs is None:
        runs = len(targets)
    runs = min(runs, len(targets))

    rng = np.random.default_rng(42)
    sampled_targets = rng.choice(targets, size=runs, replace=False)

    print(" " * 7 + "#" + "  Taxa")
    print("-" * 8 + "  " + "-" * 32)
    for run, target in enumerate(sampled_targets):
        print(f"{run + 1:>8}  {target}")

        X = df.drop(columns=[target])
        y = df[target].values

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        y_train_trues = []
        y_test_trues = []
        y_train_preds = []
        y_test_preds = []
        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            X_train = X.iloc[train_index, :]
            X_test = X.iloc[test_index, :]
            y_train_true = y[train_index]
            y_test_true = y[test_index]

            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train_true)

            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)

            y_train_trues.extend(y_train_true)
            y_test_trues.extend(y_test_true)
            y_train_preds.extend(y_train_pred)
            y_test_preds.extend(y_test_pred)

        r2_train = r2_score(y_train_trues, y_train_preds)
        r2_test = r2_score(y_test_trues, y_test_preds)
        results.append(
            {
                "species": target,
                "r2_train": r2_train,
                "r2_test": r2_test,
            }
        )
        # print("Models' average R² across all folds:")
        # print(f"R² train = {r2_train:.4f}")
        # print(f"R² test = {r2_test:.4f}")
    print("DONE")

    print("Writing results...", end="")
    results = pd.DataFrame(results)
    print("\nOverall R² across all species:")
    train_mean = np.mean(results[["r2_train"]].values)
    train_std = np.std(results[["r2_train"]].values)
    test_mean = np.mean(results[["r2_test"]].values)
    test_std = np.std(results[["r2_test"]].values)
    with open(f"taxa_prediction_{file_name}.txt", "w") as file:
        file.write(f"train: mean={train_mean:.4f} (sigma={train_std:.4f})\n")
        file.write(f"test:  mean={test_mean:.4f} (sigma={test_std:.4f})\n")
    results.to_csv(f"{file_name}.csv", index=False)
    print("DONE")


def main():
    print("-------------------")
    print("| TAXA PREDICTION |")
    print("-------------------")

    taxa_prediction(
        "Fungi",
        "taxa_prediction--fungi--2019--field_soil--control",
        years=2019,
        habitats="Field_Soil",
        beneficials="Control",
        runs=16,
    )


if __name__ == "__main__":
    main()
