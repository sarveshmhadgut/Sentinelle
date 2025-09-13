import os
import json
import yaml
import pickle
from dvclive import Live
import numpy as np
import pandas as pd
from logger import Logger
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

# from dvclive import Live
logger = Logger(name="model_evaluation", log_file="model_evaluation.log")


def load_params(params_filepath: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        params_filepath (str): The file path of the YAML configuration to load.

    Returns:
        dict: The loaded parameters as a dictionary.

    Raises:
        FileNotFoundError: If the file is not found at the given path.
        yaml.YAMLError: If the YAML file is malformed.
        Exception: For any unexpected errors.
    """
    if not isinstance(params_filepath, str):
        raise TypeError("params_path must be a string")

    try:
        logger.info(f"Loading params...")

        with open(params_filepath, "r") as f:
            params = yaml.safe_load(f)

        logger.debug(f"Params loaded successfully")
        return params

    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {params_filepath}")
        raise e

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise e


def load_model(model_filepath: str) -> object:
    """
    Load a machine learning model object from a pickle file.

    Args:
        model_filepath (str): Path to the pickle file containing the model.

    Returns:
        object: The loaded model.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For unexpected errors during model loading.
    """
    if not isinstance(model_filepath, str):
        raise TypeError("filepath must be a string")

    try:
        logger.info(f"Loading model...")

        with open(model_filepath, "rb") as f:
            model = pickle.load(f)

        logger.debug(f"Model loaded successfully")
        return model

    except FileNotFoundError as e:
        logger.error(f"File not found - {model_filepath}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise


def load_data(data_filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV URL or file path into a DataFrame.

    Args:
        data_filepath (str): URL or local path of CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        pd.errors.ParserError: Parsing errors.
        Exception: Other errors.
    """
    if not isinstance(data_filepath, str):
        raise TypeError("data_filepath must be a string")

    try:
        logger.info(f"Loading data from: {data_filepath}")
        df = pd.read_csv(data_filepath)

        logger.info(f"Data loaded successfully from {data_filepath}")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


def evaluate_model(
    classifier: Any, X_test: Any, y_test: Any
) -> Tuple[Dict[str, float], pd.DataFrame]:
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
        logger.info("Evaluating metrics...")

        y_pred = classifier.predict(X_test)
        logger.debug("Predictions generated for test set.")

        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        logger.debug("Probability predictions generated for test set.")

        accuracy = np.round(accuracy_score(y_test, y_pred), 5)
        precision = np.round(precision_score(y_test, y_pred), 5)
        recall = np.round(recall_score(y_test, y_pred), 5)
        roc_auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

        logger.debug(f"All metrics computed successfully")
        return metrics, np.round(report_df, 5)

    except Exception as e:
        logger.error(f"Unexpected error while evaluating metrics: {e}")
        raise


def save_metrics(metrics: Dict, report: pd.DataFrame, dirpath: str) -> None:
    """
    Save evaluation metrics to a JSON file and classification report as CSV.

    Args:
        metrics (Dict): Dictionary containing metric names and values.
        report (pd.DataFrame): Classification report as a DataFrame.
        dirpath (str): Directory path to save the metrics files.

    Raises:
        TypeError: If metrics or report types are incorrect, or dirpath is not a string.
        Exception: For errors during saving.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame")

    if not isinstance(dirpath, str):
        raise TypeError("dirpath must be a string")

    try:
        logger.info("Saving metrics and classification report...")

        os.makedirs(dirpath, exist_ok=True)
        report_filepath = os.path.join(dirpath, "report.csv")
        report.to_csv(report_filepath, index=True)

        metrics_filepath = os.path.join(dirpath, "metrics.json")
        with open(metrics_filepath, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics and report saved successfully")

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
        params = load_params("params.yaml")

        model_path = params["model_evaluation"]["model_filepath"]
        classifier = load_model(model_filepath=model_path)

        test_data_path = params["model_evaluation"]["test_data_path"]
        test_data = load_data(data_filepath=test_data_path)
        logger.debug("Testing data loaded successfully")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics, report = evaluate_model(
            classifier=classifier, X_test=X_test, y_test=y_test
        )

        with Live(save_dvc_exp=True) as live:
            live.log_metric("test_size", params["data_ingestion"]["test_size"])
            live.log_metric(
                "max_features", params["feature_engineering"]["max_features"]
            )
            live.log_metric("n_estimators", params["model_training"]["n_estimators"])
            live.log_metric("random_state", params["model_training"]["random_state"])
            live.log_metric("accuracy", metrics["accuracy"])
            live.log_metric("precision", metrics["precision"])
            live.log_metric("recall", metrics["recall"])
            live.log_metric("roc_auc", metrics["roc_auc"])

        metrics_dirpath = params["model_evaluation"]["metrics_dirpath"]
        save_metrics(metrics=metrics, report=report, dirpath=metrics_dirpath)

        logger.debug("Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error in model evaluation pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
