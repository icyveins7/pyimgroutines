"""
Demonstrate lazy PgFigure construction with PgFigureBrowser.

Each browser entry stores only small configuration values. Selecting an entry
waits to simulate a file read, creates its image array, and returns a completely
configured PgFigure. Selecting another entry disposes the old figure and its
array before loading the replacement.
"""

from functools import partial
import time

import numpy as np
import pyqtgraph as pg

from pyimgroutines.plots import PgFigure, PgFigureBrowser, forceShow


def load_image_figure(
    tag: str,
    seed: int,
    cmapName: str,
    delaySeconds: float = 2.0,
) -> PgFigure:
    """
    Simulate loading one tagged image and return its configured figure.

    Parameters
    ----------
    tag : str
        Image tag used as the figure title.
    seed : int
        Random seed standing in for a file identifier or byte offset.
    cmapName : str
        PyQtGraph colormap name associated with this image.
    delaySeconds : float
        Artificial loading delay in seconds.

    Returns
    -------
    PgFigure
        Newly constructed figure containing only the selected image.
    """
    # The delay represents opening a file and reading only the selected image.
    print(f"Loading {tag!r}...")
    time.sleep(delaySeconds)

    # The array is created here, not while the browser entries are registered.
    rng = np.random.default_rng(seed)
    image = rng.normal(loc=seed, scale=1.0, size=(768, 1024)).astype(np.float32)

    # Every factory controls the full figure configuration for its own entry.
    figure = PgFigure(title=tag)
    figure.plt.image(image, cmap=pg.colormap.get(cmapName))
    figure.plt.cbar.setLevels(np.percentile(image, (2.0, 98.0)))
    figure.plt.setAspectLocked()
    figure.plt.setTitle(f"{tag} — {cmapName}")
    return figure


# Factories capture only lightweight tags and loading parameters, not image arrays.
browser = PgFigureBrowser(title="Lazy image browser")
browser.addFigure(
    "reference/0001",
    partial(load_image_figure, "reference/0001", 1, "viridis"),
)
browser.addFigure(
    "reference/0002",
    partial(load_image_figure, "reference/0002", 2, "plasma"),
)
browser.addFigure(
    "reference/0003",
    partial(load_image_figure, "reference/0003", 3, "inferno"),
)

# Show the persistent browser first, then lazily load its initial selection.
browser.show()
browser.selectFigure(0)
forceShow()
