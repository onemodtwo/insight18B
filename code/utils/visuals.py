# coding: utf-8

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import numpy as np


def bar(series, cats, ax_labels, title, colors, width=0.3, hline=None,
        hline_width=1, hline_color='#000000', xlims=None, ylims=None,
        legend_loc='upper center', alpha=1.0, label_size=14, figsize=(11, 6),
        tick_label_size=13, tick_label_rot=45):
    n = len(cats)
    parity = (len(series) + 1) % 2
    start = -(len(series) // 2)
    # Create the plot
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=16)
    for i, ser in enumerate(series, start):
        plt.bar(np.arange(n) + (i + parity * 0.5) * width, series[ser],
                alpha=alpha, width=width, align="center", color=colors[ser],
                label=ser)
    plt.xticks(np.arange(n), cats, rotation=tick_label_rot)
    plt.tick_params(labelsize=tick_label_size)
    if xlims:
        plt.xlim(xlims)
    if ylims:
        plt.ylim(ylims)
    plt.xlabel(ax_labels[0], fontsize=label_size)
    plt.ylabel(ax_labels[1], fontsize=label_size)
    if type(hline) != type(None):
        plt.axhline(y=hline, linewidth=hline_width, color=hline_color)
    if legend_loc:
        plt.legend(loc=legend_loc)
    plt.tight_layout()
    return fig


def scatter(x, y, ax_labels, title, title_size=16, size=14, marker_style='o',
            face_color='black', edge_color='none', alpha=1.0, label_size=14,
            figsize=(11, 6), tick_label_size=13, trendline=False):
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=title_size)
    plt.plot(x, y, linestyle='none', markersize=size, marker=marker_style,
             markerfacecolor=face_color, markeredgecolor=edge_color,
             alpha=alpha)
#    plt.xlim((min(x) - 0.5, max(x) + 0.5))
    plt.tick_params(labelsize=tick_label_size)
    plt.xlabel(ax_labels[0], fontsize=label_size)
    plt.ylabel(ax_labels[1], fontsize=label_size)
    if trendline:
        t_line = np.polyfit(x, y, 1)
        y_hat = [val * t_line[0] + t_line[1] for val in x]
    plt.plot(x, y_hat, 'k--')
    plt.tight_layout()
    return fig


def heatmap(data, ax_labels, title, vmin=0, vmax=1, title_size=15,
            label_size=13, figsize=(8, 8), cmap='afmhot', color_bar=True):
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=title_size)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(range(len(data.columns)), ax_labels[0], fontsize=label_size)
    plt.yticks(range(len(data.index)), ax_labels[1], fontsize=label_size)
    if color_bar:
        plt.colorbar()
    return fig
