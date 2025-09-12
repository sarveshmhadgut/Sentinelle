import os
import nltk
import string
import pandas as pd
from logger import Logger
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")

logger = Logger(name="data_preprocessing", log_file="data_preprocessing.log")


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
        logger.info(
            f"Preprocessing started: transforming '{text_col}' & encoding '{target_col}'"
        )

        initial_shape = df.shape
        df = df.drop_duplicates(keep="first")
        logger.debug(
            f"Removed duplicates: {df.shape[0]} rows remaining ({initial_shape[0]} initially)"
        )

        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logger.debug(f"Text transformation completed for column '{text_col}'")

        le = LabelEncoder()
        df.loc[:, target_col] = le.fit_transform(df[target_col])

        logger.debug("Preprocessing completed successfully.")
        return df

    except KeyError as e:
        logger.error(f"Missing expected column during preprocessing: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise


def main(text_col: str = "text", target_col: str = "target") -> None:
    """
    Load raw train and test data, preprocess it, and save the processed output files.

    Args:
        text_col (str): Name of the text column in the data.
        target_col (str): Name of the target column in the data.

    Raises:
        FileNotFoundError: If the raw data files are missing.
        pd.errors.EmptyDataError: If input files are empty.
        Exception: For other unexpected errors.
    """
    if not isinstance(text_col, str) or not isinstance(target_col, str):
        raise TypeError("text_col and target_col must be strings")

    try:
        logger.info("Data preprocessing pipeline started")
        raw_data_path = "./data/raw/"
        train_path = raw_data_path + "train.csv"
        test_path = raw_data_path + "test.csv"

        train_data = pd.read_csv(train_path)
        logger.debug(f"Training data loaded with shape: {train_data.shape}")

        test_data = pd.read_csv(test_path)
        logger.debug(f"Testing data loaded with shape: {test_data.shape}")

        train_processed_data = preprocess(
            df=train_data, text_col=text_col, target_col=target_col
        )
        logger.debug("Training data preprocessed")

        test_processed_data = preprocess(
            df=test_data, text_col=text_col, target_col=target_col
        )
        logger.debug("Testing data preprocessed")

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_path = os.path.join(data_path, "train_processed.csv")
        test_processed_path = os.path.join(data_path, "test_processed.csv")

        train_processed_data.to_csv(train_processed_path, index=False)
        logger.debug(f"Processed training data saved to {train_processed_path}")

        test_processed_data.to_csv(test_processed_path, index=False)
        logger.debug(f"Processed testing data saved to {test_processed_path}")

        logger.info("Data preprocessing pipeline completed successfully.")

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
