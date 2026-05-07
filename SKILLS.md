---
description: How to use the pyimgroutines library for interactive image plotting
---

# pyimgroutines

A Python library built on **pyqtgraph** + **PySide6** for fast, interactive image visualization.

## Installation

```bash
pip install -e <path_to_pyimgroutines_repo>
```

Requires Python >= 3.9. Key dependencies: `pyqtgraph`, `PySide6`, `numpy`, `scipy`, `matplotlib`.

---

## Plotting with `PgFigure`

`PgFigure` is the main plotting class. It wraps a `QMainWindow` with a `GraphicsLayoutWidget` and provides `PgPlotItem` subplots.

### Single image

```python
from pyimgroutines.plots import PgFigure, forceShow
import numpy as np

data = np.random.rand(256, 256).astype(np.float32)

fig = PgFigure(title="My Image")
fig.plt.image(data)                   # viridis colormap, auto colorbar
fig.plt.cbar.setLevels((0.3, 0.7))    # adjust colorbar range
fig.plt.setAspectLocked()             # equal aspect ratio
fig.show()
forceShow()  # required outside ipython; blocks until window closes
```

In **ipython**, run `%gui qt` first, then you can omit `forceShow()`.

### Multiple subplots

```python
fig = PgFigure()
fig.setPlotGrid(2, 3, linkX=True, linkY=True, aspectLocked=True)
fig[0, 0].image(img1)
fig[0, 1].image(img2)
fig[1, 2].image(img3)
fig.show()
forceShow()
```

### `image()` key parameters

| Parameter | Default | Description |
|---|---|---|
| `arr` | *(required)* | 2D numpy array |
| `xywh` | `None` | `[x, y, w, h]` bounding rect; defaults to pixel coords |
| `xMesh_yMesh` | `None` | Tuple of `(xMesh, yMesh)` ndarrays; auto-computes `xywh` |
| `levels` | `None` | `(min, max)` for colorbar; `None` = auto |
| `addHalfPixelBorder` | `True` | Centre pixels on grid coords |
| `cmap` | `viridis` | `pg.colormap.ColorMap` or `None` for grayscale |
| `colorbar` | `True` | Show colorbar |
| `includeLegend` | `False` | Add legend (for overlaid line plots) |

### Drawing overlays

```python
# Rectangles
fig.plt.rectangle((x, y), (w, h), pen="r")

# Ellipses (each row: [cx, cy, rx, ry])
fig.plt.ellipses(np.array([[10, 20, 5, 3]]), pen=pg.mkPen("r"))

# Circles (each row: [cx, cy, r])
fig.plt.circles(np.array([[10, 20, 5]]), pen=pg.mkPen("b"))
```

### Hotkeys (press `H` in the figure window for reference)

| Key | Action |
|---|---|
| `V` | Rotate cursor text mode: position → pixel value → none |
| `C` | Toggle cursor text colour (dark ↔ light) |
| `L` | Toggle magnetized (pixel-locked) cursor |
| `A` | Toggle aspect ratio lock |
| `I` | Toggle image visibility |
| `R` | Toggle ROI selection box |
| `T` | Toggle targeting crosshair |
| `O` | Toggle measure line |
| `B` | Toggle status bar |
| `H` | Show help dialog |
| `G` then coords then `G` | Zoom to coordinates |
| `G` then range then `C` | Change colorbar range |
| Double-click | Maximize subplot |
| `Esc` | Restore all subplots |

### Keybuffer commands (zoom-to / colorbar)

While the figure is focused, press `G` to enter keybuffer mode. The status bar shows what you've typed. Then type a numeric expression and finish with a second hotkey:

- **Zoom** (`G` → input → `G`): input is `x,y` where each coordinate can be:
  - A single number (e.g. `100`) — centres the view on that value, keeping the current span
  - A range `lo:hi` (e.g. `50:150`) — sets the axis range directly
  - Empty — keeps the current range for that axis
  - Examples: `100,200` centres on (100,200). `50:150,` sets x to [50,150] and keeps y. `,0:500` keeps x and sets y to [0,500].
- **Colorbar** (`G` → input → `C`): input is `lo:hi` where either side can be omitted to keep the current bound. E.g. `0.1:0.9` sets both, `:0.9` only changes the upper bound, `0.1:` only changes the lower bound.
- Press `Backspace` while typing to delete the last character.
