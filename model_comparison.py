from typing import Literal, Dict, Tuple, Optional
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from FFNN import evaluate_model as evaluate_ffnn

ModelType = Literal["KNN", "Lasso", "SVR", "RandomForest", "FastforwardNN"]

MODEL_CLASSES: Dict[str, BaseEstimator] = {
    "KNN": KNeighborsRegressor,
    "Lasso": Lasso,
    "SVR": SVR,
    "RandomForest": RandomForestRegressor,
    "FastforwardNN": None,
}


def get_model(model: ModelType, hparams: dict = {}) -> BaseEstimator:
    """
    Get the specified model with given hyperparameters.

    Args:
        model (ModelType): The type of model to get.
        hparams (dict): Hyperparameters for the model.

    Returns:
        BaseEstimator: An instance of the specified model.

    Raises:
        ValueError: If an invalid model type is specified.
    """
    model_class = MODEL_CLASSES.get(model)
    if model_class is None:
        raise ValueError(f"Invalid model: {model}")
    return model_class(**hparams)


def preprocess_data(data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Preprocess the data by cleaning and encoding.

    Args:
        data (pd.DataFrame, optional): Input data. If None, data is loaded from file.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Load data
    if data is None:
        data = pd.read_excel("AI_ART_data.xlsx")

    # Data cleaning
    for col in ["price_USD", "num_item_review"]:
        data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors="coerce")
    data["num_item_review"] = data["num_item_review"].fillna(0).astype(int)

    # One-hot encoding
    categorical_columns = [
        "hue",
        "categorized_main_object",
        "theme_of_artwork",
        "materials_of_artwork",
    ]
    return pd.get_dummies(data, columns=categorical_columns)


def prepare_data(
    data: pd.DataFrame, feature_type: str = "F1"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Set up the feature matrix and target vector for modeling.

    Args:
        data (pd.DataFrame): Input data.
        features (list[str], optional): List of feature names. If None, default features are used.
        target (str): Name of the target variable.

    Returns:
        tuple: Feature matrix (X) and target vector (y).
    """
    categorical_columns = [
        "hue",
        "categorized_main_object",
        "theme_of_artwork",
        "materials_of_artwork",
    ]
    baseline_features = [
        "saturation",
        "value",
        "color_complexity",
        "artwork_quality",
        "object_complexity",
        "price_USD",
        "star_seller",
        "printable",
        "customizable",
    ] + list(data.columns[data.columns.str.startswith(tuple(categorical_columns))])

    if feature_type == "F1":
        features = baseline_features
    else:
        features = baseline_features + [
            "adult_content_score",
            "aesthetic_score",
            "image_text_similarity",
            "artwork_sentiment",
        ]

    X = data[features]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, features


def evaluate_model(
    model: ModelType,
    X: pd.DataFrame,
    y: pd.Series,
    hparams: dict = {},
    scoring: str = "neg_mean_absolute_error",
) -> float:
    """
    Evaluate the specified model using cross-validation.

    Args:
        model (ModelType): The type of model to evaluate.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        hparams (dict): Hyperparameters for the model.
        scoring (str): Scoring metric for cross-validation.

    Returns:
        float: Mean cross-validation score.
    """

    if model == "FastforwardNN":
        return evaluate_ffnn(X, y)[0]

    model = get_model(model, hparams)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return np.abs(cv_scores.mean())


def compare_models(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Compare the performance of multiple models using cross-validation.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        models (list[ModelType]): List of models to compare.
        hparams (dict): Hyperparameters for the models.

    Returns:
        pd.DataFrame: DataFrame of model scores.
    """
    hparams = {"kernel": "linear"}

    model_scores = {}
    for model in MODEL_CLASSES:
        if model == "SVR":
            model_scores[model] = evaluate_model(model, X, y, hparams)
        else:
            model_scores[model] = evaluate_model(model, X, y)

    return model_scores
