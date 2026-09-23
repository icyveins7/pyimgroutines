from pyimgroutines.plots import PgFigure, closeAllFigs, forceShow
closeAllFigs()

from pyimgroutines.plots.customitems import RecolorableLegendItem

fig = PgFigure()
fig.plt.addLegend()
p1 = fig.plt.plot([1,2,3,2,3], name="test")


fig.show()
forceShow()
