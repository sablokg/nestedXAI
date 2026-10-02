"""
Stacked-ensemble classifier (EBM + XGBoost + LightGBM + CatBoost -> Logistic Regression)
with cross-validated metrics, exposed as a Click CLI.

Usage:
    python machinelearn.py path/to/data.csv
    python machinelearn.py path/to/data.csv --target label --folds 10
"""

import click
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def build_stack(random_state: int = 42) -> StackingClassifier:
    ebm = ExplainableBoostingClassifier(
        interactions=10,
        random_state=random_state,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=random_state,
        eval_metric="logloss",
    )
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
    )
    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        random_seed=random_state,
    )

    return StackingClassifier(
        estimators=[
            ("ebm", ebm),
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("cat", cat),
        ],
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )


def machinelearn(
    csv: str,
    target: str = "target",
    folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
):
    df = pd.read_csv(csv)

    if target not in df.columns:
        raise click.ClickException(
            f"Target column '{target}' not found in {csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # X = every column except the target; y = the target column
    X = df.drop(columns=[target])
    y = df[target]

    stack = build_stack(random_state=random_state)

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        stack,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    click.echo("")
    for metric in scoring:
        scores = results[f"test_{metric}"]
        click.echo(f"{metric:10s}: {scores.mean():.4f} ± {scores.std():.4f}")
    click.echo("")

    return results


@click.command()
@click.argument("csv", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--target",
    default="target",
    show_default=True,
    help="Name of the target/label column in the CSV.",
)
@click.option(
    "--folds",
    default=5,
    show_default=True,
    type=int,
    help="Number of stratified CV folds.",
)
@click.option(
    "--random-state",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--n-jobs",
    default=-1,
    show_default=True,
    type=int,
    help="Parallel jobs for cross_validate (-1 = all cores).",
)
def cli(csv, target, folds, random_state, n_jobs):
    """Run a stacked-ensemble classifier with cross-validated metrics on CSV."""
    machinelearn(
        csv=csv,
        target=target,
        folds=folds,
        random_state=random_state,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    cli()