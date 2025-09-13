import os
import yaml
import pandas as pd
from logger import Logger
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

logger = Logger(name="feature_engineering", log_file="feature_engineering.log")


def load_params(params_filepath: str) -> Dict[str, Any]:
    """
    Load parameters from a YAML file.

    Args:
        params_filepath (str): Path to the YAML parameter file.

    Returns:
        Dict[str, Any]: Dictionary containing parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If YAML parsing fails.
        Exception: For other errors.
    """
    if not isinstance(params_filepath, str):
        raise TypeError("params_path must be a string")

    try:
        logger.info(f"Loading params...")
        with open(params_filepath, "r") as f:
            params = yaml.safe_load(f)
        logger.debug(f"Params loaded successfully")
        return params

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_filepath}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
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


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
):
    """
    Apply TF-IDF vectorization on train and test text data and return DataFrames with features and labels.

    Args:
        train_data (pd.DataFrame): Training dataset with columns 'text' and 'target'.
        test_data (pd.DataFrame): Testing dataset with columns 'text' and 'target'.
        max_features (int): Maximum number of features (vocabulary size) to consider.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with TF-IDF features and 'label' column for train and test sets.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If max_features is not positive.
        Exception: For unexpected errors during vectorization or DataFrame creation.
    """
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data must be a pandas DataFrame")

    if not isinstance(max_features, int) or max_features <= 0:
        raise ValueError("max_features must be a positive integer")

    try:
        logger.debug("Applying TF-IDF...")
        train_data.fillna("", inplace=True)
        test_data.fillna("", inplace=True)
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data["text"].values
        y_train = train_data["target"].values

        X_test = test_data["text"].values
        y_test = test_data["target"].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
        train_tfidf_df["label"] = y_train

        test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray())
        test_tfidf_df["label"] = y_test

        logger.debug(f"TF-IDF vectorization successfully applied")

        return train_tfidf_df, test_tfidf_df

    except Exception as e:
        logger.error(f"Error while applying TF-IDF vectorization: {e}")
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_dirpath: str
) -> None:
    """
    Save train and test data to CSV files under specified path.

    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        data_dirpath (str): Directory to save files.

    Raises:
        Exception: Saving errors.
    """
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data must be a pandas DataFrame")

    if not isinstance(data_dirpath, str):
        raise TypeError("data_path must be a string")

    try:
        os.makedirs(data_dirpath, exist_ok=True)
        logger.info(f"Saving training and testing data to: {data_dirpath}")

        train_data_filepath = os.path.join(data_dirpath, "tfidf_train.csv")
        test_data_filepath = os.path.join(data_dirpath, "tfidf_test.csv")

        train_data.to_csv(train_data_filepath, index=False)
        test_data.to_csv(test_data_filepath, index=False)

        logger.debug(f"Training and testing data saved successfully at {data_dirpath}")

    except Exception as e:
        logger.error(f"Unexpected error saving data: {e}")
        raise


def main() -> None:
    """
    Main pipeline function to load processed data, apply TF-IDF vectorization,
    and save the resulting feature sets.

    Raises:
        Exception: Propagates any exception that occurs during the process.
    """
    try:
        logger.info("Feature engineering pipeline started")
        params = load_params("params.yaml")["feature_engineering"]

        preprocessed_data_dirpath = params["preprocessed_data_dirpath"]

        train_preprocessed_data_path = os.path.join(
            preprocessed_data_dirpath, "preprocessed_train.csv"
        )
        test_preprocessed_data_path = os.path.join(
            preprocessed_data_dirpath, "preprocessed_test.csv"
        )

        train_preprocessed_data = load_data(data_filepath=train_preprocessed_data_path)
        test_preprocessed_data = load_data(data_filepath=test_preprocessed_data_path)

        train_tfidf_df, test_tfidf_df = apply_tfidf(
            train_data=train_preprocessed_data,
            test_data=test_preprocessed_data,
            max_features=params["max_features"],
        )

        save_data(
            train_data=train_tfidf_df,
            test_data=test_tfidf_df,
            data_dirpath=params["tfidf_data_dirpath"],
        )

        logger.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
