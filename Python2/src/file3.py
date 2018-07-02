import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

py.init_notebook_mode(connected=True)

trace1 = go.Scatter(
    x=np.random.randint(-10,10,100),
    y=np.random.randint(-10,10,100)
)
trace2 = go.Scatter(
    x=np.random.randint(-10,10,100),
    y=np.random.randint(-10,10,100)
)
py.plot([trace1, trace2])
