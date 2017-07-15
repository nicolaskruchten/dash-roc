# -*- coding: utf-8 -*-
import dash, dash_core_components as dcc, dash_html_components as html
import plotly.graph_objs as go
import math, pandas as pd, numpy as np


n = 1000
random_scores = np.random.normal(0,1,2*n)
posneg = np.repeat([1,-1], n)
def generate_roc_curve_df(goodness, skew):
    df = pd.DataFrame(dict(
        pos = (1+posneg)/2, neg = (1-posneg)/2,
        score = random_scores*(2+posneg*skew)+posneg*goodness
    )).sort_values(by='score', ascending=False)
    df["tp"] = df["pos"].cumsum()
    df["fp"] = df["neg"].cumsum()
    df["tn"] = n-df["pos"].cumsum()
    df["fn"] = n-df["neg"].cumsum()
    df["tpr"] = df["tp"]/n
    df["fpr"] = df["fp"]/n
    df["tnr"] = df["tn"]/n
    df["fnr"] = df["fn"]/n
    return df

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        html.Label('Goodness'),
        dcc.Slider(id='goodness', min=0.0, max=5.0, value=2.5, step=0.5),
        html.Label('Skew'),
        dcc.Slider(id='skew', min=-0.4, max=0.4, value=0.0, step=0.1),
        html.Label('Threshold'),
        dcc.Slider(id='threshold', min=0, max=2*n, value=n, step=n/10),
        ], style={'columnCount': 3}),
    html.Div(id="charts")
])

@app.callback(
    dash.dependencies.Output('charts', 'children'),
    [dash.dependencies.Input('goodness', 'value'),
     dash.dependencies.Input('skew', 'value'),
     dash.dependencies.Input('threshold', 'value')])
def update_charts(goodness, skew, threshold):
    df = generate_roc_curve_df(goodness, skew)

    hist_kwargs = dict(
        opacity=0.75, autobinx=False,
        xbins=dict( start=-10.0, end=10, size=0.5 )
    )

    return [
        html.Div([
            dcc.Graph(id='classes', figure=dict(
                data = [
                    go.Histogram(x=df[df.pos==1].score, **hist_kwargs),
                    go.Histogram(x=df[df.neg==1].score, **hist_kwargs)
                ],
                layout = go.Layout(
                    barmode='overlay',
                    xaxis=dict( range=[-10, 10] ),
                    yaxis=dict( range=[0, 200] )
                )
            )),
            dcc.Graph(id='rates', figure=dict(
                data = [
                    go.Scatter(x=df.score, y=df.tpr),
                    go.Scatter(x=df.score, y=df.fpr),
                ],
                layout = go.Layout(
                    xaxis=dict( range=[-10, 10] ),
                    yaxis=dict( range=[0, 1] )
                )
            )),
            ], style={'columnCount': 2}),
        html.Div([
            dcc.Graph(id='roc', figure=dict(
                data = [
                    go.Scatter(x=df.fpr, y=df.tpr)
                ],
                layout = go.Layout(
                    xaxis=dict( range=[0, 1] ),
                    yaxis=dict( range=[0, 1] )
                )
            )),
            dcc.Graph(id='confusion', figure=dict(
                data = [
                    go.Pie(labels=['tp','fp','tn','fn'],
                        values=df.iloc[threshold][['tp','fp','tn','fn']],
                        sort=False)
                ]
            )),
            ], style={'columnCount': 2})
        ]


if __name__ == '__main__':
    app.run_server(debug=True)
