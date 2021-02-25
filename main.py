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
    help="A readable file."
)
PathWArgument = typer.Argument(
    ...,
    exists=False,
    file_okay=False,
    dir_okay=False,
    writable=True,
    readable=True,
    resolve_path=True,
    help="A file to write to (that doesn't exist yet)"
)


AMOUNT_TEST_USERS = 5
SEED = 5


def run(alg: Algorithm, X: scipy.sparse.csr_matrix, top_k: int = 5):
    """ Train a model and show the recommendations for random users. """
    random.seed(SEED)
    np.random.seed(SEED)

    alg.fit(X)
    show_recommendations(alg, X, top_k=top_k)


def run_sg(alg: Algorithm, X: scipy.sparse.csr_matrix, test_users: int = 1000, perc_history: float = 0.8, top_k: int = 5):
    """ Train a model and calculate recall@k using strong generalization. """
    from src.cross_validation.strong_generalization import strong_generalization
    from src.metric.recall import recall_k

    random.seed(SEED)
    np.random.seed(SEED)

    Xtrain, Xval_in, Xval_out = strong_generalization(X, test_users=test_users, perc_history=perc_history)
    alg.fit(Xtrain)

    predictions = alg.predict(Xval_in)
    recall_scores = recall_k(predictions, Xval_out, top_k)
    avg_recall = np.average(recall_scores)
    print(f"Average Recall@{top_k} over {Xval_in.shape[0]} users:", np.around(avg_recall, decimals=5))


def show_recommendations(alg: Algorithm, X: scipy.sparse.csr_matrix, top_k: int = 5):
    """ Show the recommendations for random users. """
    random.seed(SEED)
    np.random.seed(SEED)

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
    """ Train and predict with the Item KNN model. """
    from src.algorithm.item_knn import ItemKNN

    alg = ItemKNN(k=k, normalize=normalize)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    run(alg, X, top_k=top_k)


@app.command()
def iknn_save(path: Path = PathArgument, model: Path = PathWArgument, item_col: str = "movieId", user_col: str = "userId",
         k: int = 200, normalize: bool = False):
    """ Train the Item KNN model and save it to file. """
    from src.algorithm.item_knn import ItemKNN

    alg = ItemKNN(k=k, normalize=normalize)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    alg.fit(X).save(model)


@app.command()
def iknn_load(path: Path = PathArgument, model: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
         k: int = 200, normalize: bool = False):
    """ Load an Item KNN model from file and show predictions. """
    from src.algorithm.item_knn import ItemKNN

    alg = ItemKNN(k=k, normalize=normalize).load(model)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)

    show_recommendations(alg, X, top_k=top_k)


@app.command()
def pop(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5):
    """ Train and predict the popularity model. """
    from src.algorithm.popularity import Popularity

    alg = Popularity()
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    run(alg, X, top_k=top_k)


@app.command()
def ease(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
         l2: float = 200.0):
    """ Train and predict with the EASE model. """
    from src.algorithm.ease import EASE

    alg = EASE(l2=l2)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    run(alg, X, top_k=top_k)


@app.command()
def ease_sg(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
            l2: float = 200.0, test_users: int = 1000, perc_history: float = 0.8):
    """ Train and predict with the EASE model using strong generalization. """
    from src.algorithm.ease import EASE

    random.seed(SEED)
    np.random.seed(SEED)

    alg = EASE(l2=l2)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    run_sg(alg, X, test_users=test_users, perc_history=perc_history, top_k=top_k)


@app.command()
def iknn_sg(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
            k: int = 200, normalize: bool = False, test_users: int = 1000, perc_history: float = 0.8):
    """ Train and predict with the EASE model using strong generalization. """
    from src.algorithm.item_knn import ItemKNN

    random.seed(SEED)
    np.random.seed(SEED)

    alg = ItemKNN(k, normalize)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    run_sg(alg, X, test_users=test_users, perc_history=perc_history, top_k=top_k)


@app.command()
def wmf(path: Path = PathArgument, item_col: str = "movieId", user_col: str = "userId", top_k: int = 5,
         alpha: float = 40.0, factors: int = 20, regularization: float = 0.01, iterations: int = 20):
    """ Train and predict with the WMF model. """
    from src.algorithm.wmf import WMF

    alg = WMF(alpha=alpha, num_factors=factors, regularization=regularization, iterations=iterations)
    X = util.path_to_csr(path, item_col=item_col, user_col=user_col)
    run(alg, X, top_k=top_k)


if __name__ == "__main__":
    app()
