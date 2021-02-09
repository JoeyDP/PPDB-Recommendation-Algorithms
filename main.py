from pathlib import Path
import random
import numpy as np
import scipy.sparse

import typer

import src.util as util
from src.algorithm.algorithm import Algorithm


app = typer.Typer()
PathArgument = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    writable=False,
    readable=True,
    resolve_path=True,
)

AMOUNT_TEST_USERS = 5
SEED = 5


def run(alg: Algorithm, X: scipy.sparse.csr_matrix, top_k: int = 5):
    random.seed(SEED)
    np.random.seed(SEED)
    alg.fit(X)

    test_users = random.sample(list(range(X.shape[0])), AMOUNT_TEST_USERS)
    test_histories = X[test_users, :]
    predictions = alg.predict(test_histories)
    recommendations, scores = util.predictions_to_recommendations(predictions, top_k=top_k)
    for index, u in enumerate(test_users):
        print("User", u)
        print("History", np.where(test_histories[index].toarray().flatten())[0])
        print("recommendations", recommendations[index])
        print("scores", scores[index])
        print()


@app.command()
def iknn(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
         k: int = 200, normalize: bool = False):
    from src.algorithm.item_knn import ItemKNN

    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    alg = ItemKNN(k=k, normalize=normalize)
    run(alg, X, top_k=top_k)


@app.command()
def pop(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5):
    from src.algorithm.popularity import Popularity

    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    alg = Popularity(k=top_k)
    run(alg, X, top_k=top_k)


@app.command()
def ease(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
         l2: float = 200.0):
    from src.algorithm.ease import EASE

    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    alg = EASE(l2=l2)
    run(alg, X, top_k=top_k)


if __name__ == "__main__":
    app()
