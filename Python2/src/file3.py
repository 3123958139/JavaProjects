from plotly.graph_objs import Scatter, Layout

import plotly.graph_objs as go
import plotly.offline as py


py.init_notebook_mode(connected=True)

trace1 = go.Scatter(
    x=[1, 2],
    y=[1, 2]
)
trace2 = go.Scatter(
    x=[1, 2],
    y=[2, 1]
)
py.iplot([trace1, trace2])
