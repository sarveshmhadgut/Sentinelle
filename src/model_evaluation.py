import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from logger import Logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# from dvclive import Live
logger = Logger(name="model_evaluation", log_file="model_evaluation.log")


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): The file path of the YAML configuration to load.

    Returns:
        dict: The loaded parameters as a dictionary.

    Raises:
        FileNotFoundError: If the file is not found at the given path.
        yaml.YAMLError: If the YAML file is malformed.
        Exception: For any unexpected errors.
    """
    if not isinstance(params_path, str):
        raise TypeError("params_path must be a string")

    try:
        logger.info(f"Loading params...")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        logger.debug(f"Parameters loaded successfully")
        return params

    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {params_path}")
        raise e

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise e


def load_model(filepath: str) -> object:
    """
    Load a machine learning model object from a pickle file.

    Args:
        filepath (str): Path to the pickle file containing the model.

    Returns:
        object: The loaded model.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For unexpected errors during model loading.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    try:
        logger.info(f"Loading model...")

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.debug(f"Model loaded successfully")
        return model

    except FileNotFoundError as e:
        logger.error(f"File not found - {filepath}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        pd.errors.ParserError: If parsing the CSV fails.
        pd.errors.EmptyDataError: If the CSV contains no data.
        Exception: For other unexpected errors.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    try:
        logger.debug(f"Loading data...")
        df = pd.read_csv(filepath)

        logger.debug(f"Data loaded successfully")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {filepath}: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV is empty- {filepath}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


from typing import Any, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(classifier: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a classification model using common metrics.

    Args:
        classifier (Any): Fitted classifier with predict and predict_proba methods.
        X_test (Any): Test feature data (array-like or DataFrame).
        y_test (Any): True test labels (array-like or Series).

    Returns:
        Dict[str, float]: Dictionary with metrics 'accuracy', 'precision', 'recall', and 'roc_auc'.

    Raises:
        Exception: If evaluation fails.
    """
    try:
        logger.info("Computing metrics")

        y_pred = classifier.predict(X_test)
        logger.debug("Predictions generated for test set.")

        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        logger.debug("Probability predictions generated for test set.")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

        logger.debug(f"All metrics computed successfully")
        return metrics

    except Exception as e:
        logger.error(f"Unexpected error while evaluating metrics: {e}")
        raise


def save_metrics(metrics: Dict, filepath: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (Dict): Dictionary containing metric names and values.
        filepath (str): Destination JSON file path.

    Raises:
        TypeError: If metrics is not a dictionary or filepath is not a string.
        Exception: For any error during saving.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    try:
        logger.info("Saving metrics...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        logger.debug(f"Metrics saved successfully")

    except Exception as e:
        logger.error(f"Unexpected error while saving metrics: {e}")
        raise


def main() -> None:
    """
    Main pipeline function for loading a trained model, making predictions on test data,
    evaluating metrics, and saving the results to disk.

    Raises:
        Exception: Propagates unexpected errors during pipeline execution.
    """
    try:
        logger.info("Model evaulation pipeline started")

        model_path = "./models/model.pkl"
        classifier = load_model(filepath=model_path)
        logger.debug("Model loaded successfully")

        test_tfidf_data_path = "./data/processed/test_tfidf.csv"
        test_data = load_data(filepath=test_tfidf_data_path)

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logger.debug(
            f"Test features and labels extracted [X_test, {X_test.shape}, y_test, {y_test.shape}]"
        )

        metrics = evaluate_model(classifier=classifier, X_test=X_test, y_test=y_test)

        metrics_filepath = "./reports/metrics.json"
        logger.debug(f"Saving metrics to {metrics_filepath}")
        save_metrics(metrics=metrics, filepath=metrics_filepath)

        logger.debug("Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error in model evaluation pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
