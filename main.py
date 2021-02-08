from pathlib import Path

import typer

from src.item_knn import ItemKNN


PathArgument = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    )

app = typer.Typer()


@app.command()
def iknn(path: Path = PathArgument):
    print(path)


@app.command()
def uknn(path: Path = PathArgument):
    print(path)


if __name__ == "__main__":
    app()
