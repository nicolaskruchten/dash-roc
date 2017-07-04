# -*- coding: utf-8 -*-
import dash, dash_core_components as dcc, dash_html_components as html
import plotly.graph_objs as go
import math, pandas as pd, numpy as np

def generate_roc_curve_df(goodness, skew):
    n = 10000
    df = pd.DataFrame(np.array([
        np.arange(n + n),
        np.concatenate((np.ones((n,)), np.zeros((n,)))),
        np.concatenate((np.zeros((n,)), np.ones((n,)))),
        np.concatenate((
                        np.random.normal(0+goodness,goodness/2.0+goodness*skew+1.0,n),
                        np.random.normal(0-goodness,goodness/2.0-goodness*skew+1.0,n)
                    ))
    ]).T, columns=["index","pos", "neg", "score"] )
    df = df.sort_values(by='score', ascending=False)
    df["tpr"] = (df["pos"].cumsum())/n
    df["fpr"] = (df["neg"].cumsum())/n
    df["tnr"] = (n-df["pos"].cumsum())/n
    df["fnr"] = (n-df["neg"].cumsum())/n
    df["tp"] = (df["pos"].cumsum())
    df["fp"] = (df["neg"].cumsum())
    df["tn"] = n-(df["pos"].cumsum())
    df["fn"] = n-(df["neg"].cumsum())
    return df

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        html.Label('Goodness'),
        dcc.Slider(id='goodness', min=0.0, max=5.0, value=2.5, step=0.5),
        html.Label('Skew'),
        dcc.Slider(id='skew', min=-0.4, max=0.4, value=0.0, step=0.1),
        html.Label('Threshold'),
        dcc.Slider(id='threshold', min=0, max=20000, value=10000, step=100),
        ], style={'columnCount': 3}),
    html.Div([
        dcc.Graph(id='classes'),
        dcc.Graph(id='rates'),
        ], style={'columnCount': 2}),
    html.Div([
        dcc.Graph(id='roc'),
        dcc.Graph(id='confusion'),
        ], style={'columnCount': 2})
])

@app.callback(
    dash.dependencies.Output('classes', 'figure'),
    [dash.dependencies.Input('goodness', 'value'),
     dash.dependencies.Input('skew', 'value')])
def update_classes(goodness, skew):
    df = generate_roc_curve_df(goodness, skew)
    kwargs = dict( opacity=0.75, autobinx=False,
        xbins=dict( start=-10.0, end=10, size=0.1 )
        )
    return dict(
        data = [
            go.Histogram(x=df.query('pos==1').score, **kwargs),
            go.Histogram(x=df.query('neg==1').score, **kwargs)
        ],
        layout = go.Layout(
            barmode='overlay',
            xaxis=dict( range=[-10, 10] ),
            yaxis=dict( range=[0, 500] )
        )
    )


@app.callback(
    dash.dependencies.Output('rates', 'figure'),
    [dash.dependencies.Input('goodness', 'value'),
     dash.dependencies.Input('skew', 'value')])
def update_rates(goodness, skew):
    df = generate_roc_curve_df(goodness, skew)
    return dict(
        data = [
            go.Scatter(x=df.score, y=df.tpr),
            go.Scatter(x=df.score, y=df.fpr),
        ],
        layout = go.Layout(
            xaxis=dict( range=[-10, 10] ),
            yaxis=dict( range=[0, 1] )
        )
    )


@app.callback(
    dash.dependencies.Output('roc', 'figure'),
    [dash.dependencies.Input('goodness', 'value'),
     dash.dependencies.Input('skew', 'value')])
def update_roc(goodness, skew):
    df = generate_roc_curve_df(goodness, skew)
    return dict(
        data = [
            go.Scatter(x=df.fpr, y=df.tpr)
        ],
        layout = go.Layout(
            xaxis=dict( range=[0, 1] ),
            yaxis=dict( range=[0, 1] )
        )
    )

@app.callback(
    dash.dependencies.Output('confusion', 'figure'),
    [dash.dependencies.Input('goodness', 'value'),
     dash.dependencies.Input('skew', 'value'),
     dash.dependencies.Input('threshold', 'value')])
def update_confusion(goodness, skew, threshold):
    df = generate_roc_curve_df(goodness, skew)
    labels = ['tp', 'fp', 'tn', 'fn']
    return dict(
        data = [
            go.Pie(labels=labels, values=df.iloc[threshold][labels], sort=False)
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)
