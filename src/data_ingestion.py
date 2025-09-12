import os
import yaml
import pandas as pd
from logger import Logger
from typing import Dict, Any
from sklearn.model_selection import train_test_split

logger = Logger(name="data_ingestion", log_file="data_ingestion.log")


def load_params(params_path: str) -> Dict[str, Any]:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML parameter file.

    Returns:
        Dict[str, Any]: Dictionary containing parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If YAML parsing fails.
        Exception: For other errors.
    """
    if not isinstance(params_path, str):
        raise TypeError("params_path must be a string")

    try:
        logger.info(f"Loading params from: {params_path}")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug(f"Parameters loaded successfully")
        return params

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV URL or file path into a DataFrame.

    Args:
        data_url (str): URL or local path of CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        pd.errors.ParserError: Parsing errors.
        Exception: Other errors.
    """
    if not isinstance(data_url, str):
        raise TypeError("data_url must be a string")

    try:
        logger.info(f"Loading data from: {data_url}")

        df = pd.read_csv(data_url)

        logger.debug(f"Data loaded successfully")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw DataFrame.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.

    Raises:
        KeyError: Missing expected column.
        Exception: Other errors.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    try:
        logger.info("Preprocessing data")

        df.dropna(axis=1, inplace=True)
        df = pd.DataFrame({"text": df["v2"], "target": df["v1"]})

        logger.debug(f"Data preprocessing completed successfully")
        return df

    except KeyError as e:
        logger.error(f"Missing expected column during preprocessing: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """
    Save train and test data to CSV files under specified path.

    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        data_path (str): Directory to save files.

    Raises:
        Exception: Saving errors.
    """
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data must be a pandas DataFrame")

    if not isinstance(data_path, str):
        raise TypeError("data_path must be a string")

    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        logger.info(f"Saving data to: {raw_data_path}")

        train_data_path = os.path.join(raw_data_path, "train.csv")
        test_data_path = os.path.join(raw_data_path, "test.csv")

        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)

        logger.debug(f"Data saved successfully")

    except Exception as e:
        logger.error(f"Unexpected error saving data: {e}")
        raise


def main() -> None:
    """
    Perform data ingestion workflow.

    Raises:
        Exception: Any error during workflow.
    """
    try:
        logger.info("Data ingestion pipeline started")

        data_url: str = (
            "https://raw.githubusercontent.com/sarveshmhadgut/DSBDAL-Coursework/"
            "refs/heads/main/datasets/spam.csv"
        )

        df: pd.DataFrame = load_data(data_url=data_url)
        processed_df: pd.DataFrame = preprocess(df)

        test_size: float = 0.2
        random_state: int = 42

        train_data, test_data = train_test_split(
            processed_df, test_size=test_size, random_state=random_state
        )
        logger.debug(
            f"Split data into train ({train_data.shape[0]}) and test ({test_data.shape[0]}) samples"
        )

        save_data(train_data=train_data, test_data=test_data, data_path="./data")
        logger.info("Data ingestion process completed successfully")

    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
