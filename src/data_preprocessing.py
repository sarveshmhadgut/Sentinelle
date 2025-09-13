import os
import nltk
import yaml
import string
import pandas as pd
from logger import Logger
from typing import Dict, Any
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")

logger = Logger(name="data_preprocessing", log_file="data_preprocessing.log")


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


def transform_text(text: str) -> str:
    """
    Preprocess a text string by lowercasing, tokenizing, removing non-alphanumeric tokens,
    filtering stopwords and punctuation, and stemming the tokens.

    Args:
        text (str): The input text string to transform.

    Returns:
        str: The transformed text.

    Raises:
        TypeError: If input text is not a string.
        Exception: For any unexpected errors during processing.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")

    try:
        english_stopwords = stopwords.words("english")
        ps = PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [
            word
            for word in text
            if word not in english_stopwords and word not in string.punctuation
        ]
        text = [ps.stem(word) for word in text]

        return " ".join(text)
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise


def preprocess(
    df: pd.DataFrame, text_col: str = "text", target_col: str = "target"
) -> pd.DataFrame:
    """
    Preprocess the DataFrame by dropping duplicates, transforming the text column and encoding the target column.

    Args:
        df (pd.DataFrame): Input DataFrame containing data.
        text_col (str): Name of the text column to transform.
        target_col (str): Name of the target column to encode.

    Returns:
        pd.DataFrame: The preprocessed dataframe.

    Raises:
        KeyError: If expected columns are missing.
        Exception: For unexpected errors.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(text_col, str) or not isinstance(target_col, str):
        raise TypeError("text_col and target_col must be strings")

    try:
        logger.info(f"Starting preprocessing...")

        initial_shape = df.shape
        df = df.drop_duplicates(keep="first")
        logger.debug(
            f"Removed duplicates: {df.shape[0]} rows remaining ({initial_shape[0]} initially)"
        )

        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logger.debug(f"Text transformation completed for column '{text_col}'")

        le = LabelEncoder()
        df.loc[:, target_col] = le.fit_transform(df[target_col])
        logger.debug(f"Label encoding completed for column '{target_col}'")

        logger.debug("Preprocessing completed successfully.")
        return df

    except KeyError as e:
        logger.error(f"Missing expected column during preprocessing: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
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

        train_data_filepath = os.path.join(data_dirpath, "preprocessed_train.csv")
        test_data_filepath = os.path.join(data_dirpath, "preprocessed_test.csv")

        train_data.to_csv(train_data_filepath, index=False)
        test_data.to_csv(test_data_filepath, index=False)

        logger.debug(f"Training and testing data saved successfully at {data_dirpath}")

    except Exception as e:
        logger.error(f"Unexpected error saving data: {e}")
        raise


def main() -> None:
    """
    Load raw train and test data, preprocess it, and save the processed output files.

    Raises:
        FileNotFoundError: If the raw data files are missing.
        pd.errors.EmptyDataError: If input files are empty.
        Exception: For other unexpected errors.
    """
    try:
        logger.info("Data pre-processing pipeline started")
        params = load_params(params_filepath="params.yaml")["data_preprocessing"]

        raw_data_dirpath = params["raw_data_dirpath"]
        train_filepath = os.path.join(raw_data_dirpath, "raw_train.csv")
        test_filepath = os.path.join(raw_data_dirpath, "raw_test.csv")

        train_data = load_data(data_filepath=train_filepath)
        test_data = load_data(data_filepath=test_filepath)

        text_col = params["text_col"]
        target_col = params["target_col"]

        train_preprocessed_data = preprocess(
            df=train_data, text_col=text_col, target_col=target_col
        )
        logger.debug("Training data preprocessed")

        test_preprocessed_data = preprocess(
            df=test_data, text_col=text_col, target_col=target_col
        )
        logger.debug("Testing data preprocessed")

        save_data(
            train_data=train_preprocessed_data,
            test_data=test_preprocessed_data,
            data_dirpath=params["preprocessed_data_dirpath"],
        )

        logger.info("Data pre-processing pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
