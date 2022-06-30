import os

import numpy as np
import matplotlib.pyplot as plt

from graph_globals import global_params
from graphs import graph, multi_line, get_cmap
from utils.pickle_helper import load_data

def get_headers(fp):
    with open(fp, 'r') as f:
        header = f.readline()
        headers = header.split(',')
        headers[-1] = headers[-1][:-1]
    print(headers)
    return headers

def get_data(fp):
    data = np.loadtxt(fp, delimiter=',', skiprows=1).T
    if data.ndim == 1:
        return [ [d for d in data] ]
    else:
        return [d for d in data]

def graph_data(data, labels, metric):
    f, ax = plt.subplots(1,1)
    cmap = get_cmap(len(labels))
    colours = [ cmap(i) for i in range(len(labels)) ]
    graph( ax, data, multi_line( ax, data, colours, labels),
           xtitle='Time',
           ytitle_pad = (metric, 60),
           title='Training Updates Progress',
           legend=(0.92, 0.92),
           grid=True)
    plt.show()

def graph_metric(path, metric):
    newest_fp = [fp for fp in sorted(os.listdir(path)) if metric in fp][-1]
    print(metric)
    print(newest_fp)
    fp = path+newest_fp 
    labels = get_headers(fp)
    data = get_data(fp)
    print(data)
    graph_data(data, labels, metric)

def main():
    global_params()
    path = 'tmp/'

    metrics = ['replay', 'updates', 'nexp']
    for m in metrics:
        graph_metric(path, m)


if __name__ == '__main__':
    main()
