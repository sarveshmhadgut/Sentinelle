import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from logger import Logger

logger = Logger(name="feature_engineering", log_file="feature_engineering.log")


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
        logger.info(f"Loading params from: {params_path}")
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
        logger.error(f"Unexpected error loading parameters: {e}")
        raise e


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame and fill missing values.

    Args:
        file_path (str): The file path of the CSV to load.

    Returns:
        pd.DataFrame: The loaded DataFrame with missing values filled with empty strings.

    Raises:
        pd.errors.ParserError: If there is an error parsing the CSV.
        FileNotFoundError: If the CSV file does not exist.
        Exception: For other unexpected errors.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    try:
        logger.info(f"Loading params from: {file_path}")

        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)

        logger.debug(f"Data loaded successfully")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed: {e}")
        raise

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
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
        logger.debug("Applying TF-IDF")
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data["text"].values
        y_train = train_data["target"].values

        X_test = test_data["text"].values
        y_test = test_data["target"].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logger.debug("a")

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df["label"] = y_test

        logger.debug(f"TF-IDF vectorization applied with max_features={max_features}")

        logger.debug(
            f"Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}"
        )

        return train_df, test_df

    except Exception as e:
        logger.error(f"While applying TF-IDF vectorization: {e}")
        raise


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save the given DataFrame to a CSV file at the specified filepath.
    Creates directories if they do not exist.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Destination file path for the CSV.

    Raises:
        TypeError: If input types are incorrect.
        Exception: For unexpected errors during saving.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    try:
        logger.info(f"Saving data to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.debug(f"Data saved successfully to {filepath}")

    except Exception as e:
        logger.error(f"Unexpected error while saving data to {filepath}: {e}")
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

        max_features: int = 50

        intrim_data_path: str = "./data/interim/"
        train_intrim_data_path = os.path.join(intrim_data_path, "train_processed.csv")
        test_intrim_data_path = os.path.join(intrim_data_path, "test_processed.csv")

        train_data = load_data("./data/interim/train_processed.csv")
        logger.debug(f"Read train interim data")

        test_data = load_data("./data/interim/test_processed.csv")
        logger.debug(f"Read test interim data")

        train_df, test_df = apply_tfidf(
            train_data=train_data, test_data=test_data, max_features=max_features
        )

        processed_data_path: str = "./data/processed/"
        train_processed_data_path = os.path.join(processed_data_path, "train_tfidf.csv")
        test_processed_data_path = os.path.join(processed_data_path, "test_tfidf.csv")

        save_data(df=train_df, filepath=train_processed_data_path)
        save_data(df=test_df, filepath=test_processed_data_path)

        logger.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
