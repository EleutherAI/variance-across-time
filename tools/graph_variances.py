"""
TODO Delete and replace
Script for generating graphs based on variance data.

TODO: fix legends
TODO: add option for specifying class/label versus all
"""
from argparse import ArgumentParser
import os
from itertools import product
from pathlib import Path

import plotly.graph_objs as go
import plotly as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def produce_graphs(data: dict[str, pd.DataFrame], figure_title: str) -> go.Figure:
    """Generates Figure with variance changes across training

    Args:
        data (dict[str, pd.DataFrame]): A dictionary where keys are a corruption/title and 
            each dataframe contains variances and their step
    """
    
    rgb_values = np.array([px.colors.unlabel_rgb(c) for c in px.colors.sequential.Rainbow]) / 255
    cmap = LinearSegmentedColormap.from_list('custom', rgb_values)

    # Create a subplot grid
    rows = len(data) // 4
    cols = 4
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(data.keys()),
        shared_xaxes=True,
        shared_yaxes=True
    )

    for i, title in enumerate(data):
        df = data[title]
        
        row = i // cols + 1
        col = i % cols + 1
        
        custom_colormap = [cmap(i) for i in np.linspace(0, 1, len(df.columns))]
        
        for col_i, column in enumerate(df.columns):
            color = f"rgb({custom_colormap[col_i][0]}, {custom_colormap[col_i][1]}, {custom_colormap[col_i][2]})"
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode='lines',
                    line=dict(color=color),
                    showlegend=(i == 0), 
                    name=column,
                    legendgroup=column
                ), row=row, col=col)
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')
    
    # Update the layout
    fig.update_layout(
        height=1000,
        width=1900,
        title=figure_title,
        xaxis_title='Index',
        yaxis_title='Value',
        overwrite=True
    )

    return fig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--variance-path', '-p', type=str)
    parser.add_argument('--variance-titles', '-t', nargs='+')
    parser.add_argument('--out', '-o', type=str, default='./variance_progression.png')
    parser.add_argument('--title', type=str, default="Logit Variances")
    
    args = parser.parse_args()
    
    path = args.variance_path
    titles = args.variance_titles
    output = args.out
    
    assert path is not None and titles is not None
    
    # Progress through i 1->20 steps, checking for a .csv file containing variance data.
    # We'll assume that we we see is what we get. If we get nothing, do nothing.
    variances = {title: pd.DataFrame() for title in titles}
    
    for title, step in product(titles, np.power(2, range(20))):
        file_name = f"{step}_{title}_variances.csv"
        file_path = os.path.join(path, file_name)
        
        if not os.path.isfile(file_path):
            continue
        
        data = pd.read_csv(file_path)
        variances[title].insert(0, f"Step {step}", data['all'], True)
    
    # remove empty dfs
    for title in variances:
        if len(variances[title]) == 0:
            del variances[title]
    
    # if we are left with nothing, exit
    if len(variances) == 0:
        print("No variance files found, exiting...")
        exit(0)
    
    # produce graphs
    figure = produce_graphs(variances, args.title)
    
    # save as png to out
    path_str = f'{args.out}.png' if not args.out.endswith(".png") else args.out
    
    output = Path(path_str)
    
    figure.write_image(output)
    
