import os
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from logger import Logger

logger = Logger(name="model_training", log_file="model_training.log")


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


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame, handling missing data
    and optionally skipping bad lines with warnings.

    Args:
        filepath (str): Path to the CSV file to load.

    Returns:
        pd.DataFrame: Loaded DataFrame with missing values filled as empty strings.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        pd.errors.ParserError: If there is a parsing error reading the CSV.
        Exception: For other unexpected errors.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    try:
        logger.info(f"Loading data...")

        df = pd.read_csv(filepath, on_bad_lines="warn", engine="python")
        df.fillna("", inplace=True)

        logger.debug(f"Data loaded successfully with shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model with the provided training data and parameters.

    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target vector for training.
        params (dict): Dictionary of hyperparameters including 'n_estimators' and 'random_state'.

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
        logger.info(f"Training classfier [X: {X_train.shape}, y: {y_train.shape}]...")

        classifier = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 25),
            random_state=params.get("random_state", 42),
        )
        classifier.fit(X_train, y_train)

        logger.debug("Classifier trained successfully")
        return classifier

    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while training model: {e}")
        raise


def save_model(model: object, file_path: str = "models/model.pkl") -> None:
    """
    Save a trained machine learning model to disk as a pickle file.

    Args:
        model (object): Trained ML model object to save.
        file_path (str): Destination file path for the saved model. Defaults to 'models/model.pkl'.

    Raises:
        FileNotFoundError: If the directory path does not exist and cannot be created.
        Exception: For any other errors during saving.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    try:
        logger.info(f"Saving model...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
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
        params = {"n_estimators": 25, "random_state": 42}

        train_tfidf_data_path = "./data/processed/train_tfidf.csv"
        train_data = load_data(filepath=train_tfidf_data_path)

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        classifier = train_model(X_train=X_train, y_train=y_train, params=params)
        save_model(classifier)

        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error in model training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
