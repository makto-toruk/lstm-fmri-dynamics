import numpy as np

import plotly.graph_objs as go
import plotly.express as px

def _hex_to_rgb(hex):
    '''
    for opacity of shaded error bars
    '''
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) 
        for i in range(0, hlen, hlen//3))

def _plot_ts(ts, color, showlegend=False, name=''):
    '''
    plot shaded error bars
    input:
        ts: dict {'mean': (1 x time),
            'ste: (1 x time)}
        color: hex
        name: legendname
    '''
    low = ts['mean'] - ts['ste']
    high = ts['mean'] + ts['ste']
    y = ts['mean']
    k_time = len(ts['mean'])
    x = [(ii + 1) for ii in range(k_time)]
    
    fillcolor = _hex_to_rgb(color) + (0.3,) #opacity
    fillcolor = 'rgba' + str(fillcolor)
    
    # lowerbound
    lb = go.Scatter(name=name,
        x=x, y=low,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        showlegend=False,
        legendgroup=name)
    
    # plot line
    tr = go.Scatter(name=name,
        x=x, y=y,
        mode='lines',
        line=dict(color=color),
        fillcolor=fillcolor,
        fill='tonexty',
        showlegend=showlegend,
        legendgroup=name)
    
    # upperbound
    ub = go.Scatter(name=name,
        x=x, y=high,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor=fillcolor,
        fill='tonexty',
        showlegend=False,
        legendgroup=name)

    plotter = {'lb': lb, 'tr': tr, 'ub': ub}

    return plotter

def _add_box(i, j):
    '''
    box around heatmap entry
    for highlighting
    '''
    box = {}
    line = dict(color='#00cc96', width=4)

    box['l'] = go.Scatter(x=[j-0.5, j-0.5],
        y=[i-0.5, i+0.5],
        mode='lines',
        line=line,
        showlegend=False)
    box['r'] = go.Scatter(x=[j+0.5, j+0.5],
        y=[i-0.5, i+0.5],
        mode='lines',
        line=line,
        showlegend=False)
    box['u'] =  go.Scatter(x=[j-0.5, j+0.5],
        y=[i+0.5, i+0.5],
        mode='lines',
        line=line,
        showlegend=False)
    box['d'] = go.Scatter(x=[j-0.5, j+0.5],
        y=[i-0.5, i-0.5],
        mode='lines',
        line=line,
        showlegend=False)
    
    return box

def _highlight_max(fig, z, axis=0):
    '''
    helper to highlight max in each row/col
    '''
    kk = z.shape[axis]
    for ii in range(kk):
        jj = np.argmax(z[ii])
        box = _add_box(ii, jj)
        for line in box:
            fig.add_trace(box[line])
    
    return fig

