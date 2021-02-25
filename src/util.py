from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse

ITEM_ID = 'iid'
USER_ID = 'uid'


def df_to_csr(df: pd.DataFrame):
    """ Converts a dataframe with columns ITEM_ID and USER_ID to a sparse csr matrix of interactions. """
    data = np.ones(len(df), dtype=np.int8)
    return scipy.sparse.csr_matrix((data, (df[USER_ID], df[ITEM_ID])), dtype=np.int8)


def path_to_df(path: Path, item_col, user_col):
    """ Reads a csv file from path to a pandas dataframe with columns ITEM_ID and USER_ID. """
    df = pd.read_csv(path)
    df.rename(columns={
        item_col: ITEM_ID,
        user_col: USER_ID
    }, inplace=True)
    df = df[[ITEM_ID, USER_ID]]
    return df


def path_to_csr(path: Path, item_col, user_col):
    """ Reads a csv file and converts it to a sparse csr matrix of interactions. """
    df = path_to_df(path, item_col, user_col)
    matrix = df_to_csr(df)
    return matrix


def predictions_to_recommendations(predictions: np.ndarray, top_k: int) -> np.ndarray:
    """ Takes a matrix of user-item scores and returns a ranked list of the top_k items per user. """
    recommendations = np.argpartition(predictions, -1-np.arange(top_k), axis=1)[:, -top_k:][:, ::-1]
    scores = np.take_along_axis(predictions, recommendations, axis=1)
    return recommendations, scores
