"""Small matplotlib helper utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def make_legend_togglable(ax, labels=None, **legend_kwargs):
    """
    Create a clickable legend on `ax` whose labels toggle the visibility
    of the corresponding plot handles.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes whose legend should become clickable.
    labels : list[str] | None
        Optional legend labels. If omitted, the labels already attached to
        the artists on `ax` are used.
    **legend_kwargs
        Forwarded to `ax.legend`.
    """
    handles, default_labels = ax.get_legend_handles_labels()
    if not handles:
        return

    labels = labels if labels is not None else default_labels
    legend = ax.legend(handles, labels, **legend_kwargs)
    texts = legend.get_texts()

    label_to_handle = {text.get_text(): handle for text, handle in zip(texts, handles)}

    def on_pick(event):
        text = event.artist
        handle = label_to_handle.get(text.get_text())
        if handle is None:
            return
        visible = not handle.get_visible()
        handle.set_visible(visible)
        text.set_alpha(1.0 if visible else 0.2)
        ax.figure.canvas.draw_idle()

    for text in texts:
        text.set_picker(5)
    ax.figure.canvas.mpl_connect("pick_event", on_pick)


if __name__ == "__main__":
    x = np.linspace(0, 10, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x, np.sin(x), label="sin")
    ax1.plot(x, np.cos(x), label="cos")
    make_legend_togglable(ax1, loc="upper right")

    ax2.plot(x, x**2, label="x^2")
    ax2.plot(x, x**3, label="x^3")
    make_legend_togglable(ax2, loc="upper left")

    fig.tight_layout()
    plt.show()
