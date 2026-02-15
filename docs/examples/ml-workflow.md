# ML Workflow

Build machine learning workflows with hyperparameter tuning, cross-validation, and model evaluation using Daglite.

## Overview

Machine learning workflows typically involve:

1. **Data preparation** - Load and preprocess training data
2. **Model training** - Train models with different hyperparameters
3. **Evaluation** - Assess model performance
4. **Selection** - Choose the best performing model

Daglite's `.map(..., map_mode="product")` method makes hyperparameter sweeps easy, while `.map()` and `.join()` handle parallel training and result aggregation.

## Basic ML Pipeline

### Simple Training Pipeline

```python
from daglite import task, evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@task
def load_data(path: str) -> pd.DataFrame:
    """Load training data."""
    return pd.read_csv(path)

@task
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2
) -> dict:
    """Split into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

@task
def train_model(
    data: dict,
    n_estimators: int,
    max_depth: int
) -> dict:
    """Train random forest model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(data["X_train"], data["y_train"])

    # Evaluate
    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    return {
        "model": model,
        "accuracy": accuracy,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }
    }

# Execute training pipeline
result = evaluate(
    load_data(path="train.csv")
    .then(split_data, target_col="target")
    .then(train_model, n_estimators=100, max_depth=10)
)

print(f"Model accuracy: {result['accuracy']:.3f}")
```

## Hyperparameter Tuning

### Grid Search with `.map(map_mode="product")`

```python
@task
def train_model(
    data: dict,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int
) -> dict:
    """Train model with specified hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    return {
        "model": model,
        "accuracy": accuracy,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }
    }

@task
def find_best_model(results: list[dict]) -> dict:
    """Select model with highest accuracy."""
    best = max(results, key=lambda r: r["accuracy"])
    print(f"Best accuracy: {best['accuracy']:.3f}")
    print(f"Best params: {best['params']}")
    return best

# Load and split data
data = evaluate(
    load_data(path="train.csv")
    .then(split_data, target_col="target")
)

# Hyperparameter grid search
best_model = evaluate(
    train_model.map(
        data=[data],  # Fixed parameter
        n_estimators=[50, 100, 200],
        max_depth=[5, 10, 15, None],
        min_samples_split=[2, 5, 10],
        map_mode="product"
    )
    .join(find_best_model)
)
```

### Random Search

```python
import numpy as np

@task
def generate_random_params(seed: int) -> dict:
    """Generate random hyperparameters."""
    np.random.seed(seed)
    return {
        "n_estimators": int(np.random.choice([50, 100, 150, 200])),
        "max_depth": int(np.random.choice([5, 10, 15, 20, None])),
        "min_samples_split": int(np.random.choice([2, 5, 10, 20])),
        "min_samples_leaf": int(np.random.choice([1, 2, 4, 8]))
    }

@task
def train_with_params(data: dict, params: dict) -> dict:
    """Train model with parameter dict."""
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    return {"model": model, "accuracy": accuracy, "params": params}

# Random search with 20 trials
data = evaluate(
    load_data(path="train.csv")
    .then(split_data, target_col="target")
)

best_model = evaluate(
    generate_random_params.map(seed=list(range(20)), map_mode="product")
    .then(train_with_params, data=data)
    .join(find_best_model)
)
```

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

@task
def create_folds(
    df: pd.DataFrame,
    target_col: str,
    n_folds: int = 5
) -> list[dict]:
    """Create k-fold splits."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds = []
    for train_idx, val_idx in kf.split(X):
        folds.append({
            "X_train": X.iloc[train_idx],
            "X_val": X.iloc[val_idx],
            "y_train": y.iloc[train_idx],
            "y_val": y.iloc[val_idx]
        })

    return folds

@task
def train_on_fold(
    fold: dict,
    n_estimators: int,
    max_depth: int
) -> float:
    """Train and evaluate on a single fold."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(fold["X_train"], fold["y_train"])
    y_pred = model.predict(fold["X_val"])

    return accuracy_score(fold["y_val"], y_pred)

@task
def average_scores(scores: list[float]) -> float:
    """Calculate mean cross-validation score."""
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"CV Score: {mean_score:.3f} (+/- {std_score:.3f})")
    return mean_score

# Create folds
folds = evaluate(
    load_data(path="train.csv")
    .then(create_folds, target_col="target", n_folds=5)
)

# Train on each fold
cv_score = evaluate(
    train_on_fold.map(
        fold=folds,
        n_estimators=[100],
        max_depth=[10],
        map_mode="product"
    )
    .join(average_scores)
)
```

### Cross-Validation with Hyperparameter Tuning

```python
@task
def cv_with_params(
    folds: list[dict],
    n_estimators: int,
    max_depth: int
) -> dict:
    """Run cross-validation for hyperparameter combination."""
    scores = []

    for fold in folds:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(fold["X_train"], fold["y_train"])
        y_pred = model.predict(fold["X_val"])
        score = accuracy_score(fold["y_val"], y_pred)
        scores.append(score)

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }
    }

@task
def find_best_params(results: list[dict]) -> dict:
    """Find best hyperparameters by CV score."""
    best = max(results, key=lambda r: r["mean_score"])
    print(f"Best CV score: {best['mean_score']:.3f} (+/- {best['std_score']:.3f})")
    print(f"Best params: {best['params']}")
    return best

# Create folds
folds = evaluate(
    load_data(path="train.csv")
    .then(create_folds, target_col="target", n_folds=5)
)

# Grid search with cross-validation
best_params = evaluate(
    cv_with_params.map(
        folds=[folds],
        n_estimators=[50, 100, 200],
        max_depth=[5, 10, 15],
        map_mode="product"
    )
    .join(find_best_params)
)
```

## Feature Engineering Pipeline

```python
@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
    df = df.copy()

    # Polynomial features
    df["feature1_squared"] = df["feature1"] ** 2
    df["feature1_feature2"] = df["feature1"] * df["feature2"]

    # Binning
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 50, 100],
        labels=["young", "adult", "middle", "senior"]
    )

    # One-hot encoding
    df = pd.get_dummies(df, columns=["age_group"])

    return df

@task
def select_features(
    df: pd.DataFrame,
    target_col: str,
    n_features: int = 10
) -> pd.DataFrame:
    """Select top features using mutual information."""
    from sklearn.feature_selection import mutual_info_classif

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)

    # Select top features
    top_features = X.columns[np.argsort(mi_scores)[-n_features:]]

    return df[[target_col] + list(top_features)]

# Feature engineering + training pipeline
result = evaluate(
    load_data(path="train.csv")
    .then(engineer_features)
    .then(select_features, target_col="target", n_features=10)
    .then(split_data, target_col="target")
    .then(train_model, n_estimators=100, max_depth=10)
)
```

## Model Comparison

Compare multiple model types:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

@task
def train_logistic_regression(data: dict, C: float) -> dict:
    """Train logistic regression."""
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    return {
        "model_type": "LogisticRegression",
        "model": model,
        "accuracy": accuracy,
        "params": {"C": C}
    }

@task
def train_svm(data: dict, C: float, kernel: str) -> dict:
    """Train SVM."""
    model = SVC(C=C, kernel=kernel, random_state=42)
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    return {
        "model_type": "SVM",
        "model": model,
        "accuracy": accuracy,
        "params": {"C": C, "kernel": kernel}
    }

@task
def train_gradient_boosting(
    data: dict,
    n_estimators: int,
    learning_rate: float
) -> dict:
    """Train gradient boosting."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    return {
        "model_type": "GradientBoosting",
        "model": model,
        "accuracy": accuracy,
        "params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        }
    }

@task
def compare_models(results: list[dict]) -> dict:
    """Compare all model results."""
    for result in sorted(results, key=lambda r: r["accuracy"], reverse=True):
        print(f"{result['model_type']}: {result['accuracy']:.3f} - {result['params']}")

    return max(results, key=lambda r: r["accuracy"])

# Prepare data
data = evaluate(
    load_data(path="train.csv")
    .then(split_data, target_col="target")
)

# Train all models with different hyperparameters
lr_models = train_logistic_regression.map(
    data=[data],
    C=[0.01, 0.1, 1.0, 10.0],
    map_mode="product"
)

svm_models = train_svm.map(
    data=[data],
    C=[0.1, 1.0, 10.0],
    kernel=["rbf", "linear"],
    map_mode="product"
)

gb_models = train_gradient_boosting.map(
    data=[data],
    n_estimators=[50, 100],
    learning_rate=[0.01, 0.1],
    map_mode="product"
)

# Combine and find best
@task
def combine_results(lr, svm, gb):
    return lr + svm + gb

best_model = evaluate(
    combine_results(lr=lr_models, svm=svm_models, gb=gb_models)
    .then(compare_models)
)
```

## Best Practices

### 1. Set Random Seeds

Ensure reproducibility:

```python
@task
def train_model(data: dict, seed: int = 42):
    np.random.seed(seed)
    model = RandomForestClassifier(random_state=seed)
    # ...
```

### 2. Track All Metrics

Don't just track accuracy:

```python
from sklearn.metrics import precision_score, recall_score, f1_score

@task
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }
```

### 3. Save Models

Persist trained models:

```python
import joblib

@task
def save_model(result: dict, path: str) -> str:
    """Save trained model to disk."""
    joblib.dump(result["model"], path)
    return path
```

### 4. Use Type Hints

Clear type hints improve code quality:

```python
from typing import Literal

@task
def train_model(
    data: dict,
    model_type: Literal["rf", "gb", "lr"],
    **kwargs
) -> dict:
    if model_type == "rf":
        model = RandomForestClassifier(**kwargs)
    elif model_type == "gb":
        model = GradientBoostingClassifier(**kwargs)
    elif model_type == "lr":
        model = LogisticRegression(**kwargs)

    # ...
```

## See Also

- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - ML algorithms
- [Data Processing Example](data-processing.md) - Parallel processing patterns
- [ETL Pipeline](etl-pipeline.md) - Data preparation patterns
