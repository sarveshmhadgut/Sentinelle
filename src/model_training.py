import os
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from logger import Logger

logger = Logger(name="model_training", log_file="model_training.log")


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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 25,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model with the provided training data and parameters.

    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target vector for training.
        n_estimators (int): The number of trees in the forest
        random_state (int): Controls the randomness for reproducibility

    Returns:
        RandomForestClassifier: Trained Random Forest model instance.

    Raises:
        ValueError: If the number of samples in X_train and y_train do not match.
        Exception: For unexpected errors during training.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "The number of samples in X_train and y_train must be the same."
        )

    try:
        logger.info(f"Training Random Forest classfier...")

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        classifier.fit(X_train, y_train)

        logger.debug("Random Forest classifier trained successfully")
        return classifier

    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while training model: {e}")
        raise


def save_model(model: object, model_filepath: str) -> None:
    """
    Save a trained machine learning model to disk as a pickle file.

    Args:
        model (object): Trained ML model object to save.
        model_filepath (str): Destination file path for the saved model.

    Raises:
        FileNotFoundError: If the directory path does not exist and cannot be created.
        Exception: For any other errors during saving.
    """
    if not isinstance(model_filepath, str):
        raise TypeError("file_path must be a string")

    try:
        logger.info(f"Saving model...")
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

        with open(model_filepath, "wb") as f:
            pickle.dump(model, f)

        logger.debug(f"Model saved successfully")

    except FileNotFoundError as e:
        logger.error(f"Directory not found and could not be created: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while saving model: {e}")
        raise


def main() -> None:
    """
    Main pipeline function to load processed TF-IDF features, train a Random Forest model,
    and save the trained model to disk.

    Raises:
        Exception: Propagates unexpected exceptions with logging.
    """
    try:
        logger.debug("Starting model training pipeline")

        params = load_params("params.yaml")["model_training"]

        train_data_filepath = params["train_data_filepath"]
        train_tfidf_data = load_data(data_filepath=train_data_filepath)
        logger.debug("Training data loaded successfully")

        X_train = train_tfidf_data.iloc[:, :-1].values
        y_train = train_tfidf_data.iloc[:, -1].values

        model = train_model(
            X_train=X_train,
            y_train=y_train,
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
        )

        save_model(model=model, model_filepath=params["model_filepath"])
        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error in model training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
