import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl


def graph(ax, data, graph_func, xtitle=None, ytitle_pad=None, title=None, legend=None, grid=None, xlim=None, ylim=None, colours=None):
    elements, labels = graph_func

    if xtitle:
        ax.set_xlabel(xtitle)
                                                                   
    if ytitle_pad:
        ax.set_ylabel(ytitle_pad[0], rotation=360, labelpad=ytitle_pad[1])
                                                                   
    if title:
        ax.set_title(title)

    if legend:
        if colours is None:
            ax.legend(elements, labels, loc=legend, framealpha=1.0)
        else:
            patches = [ mpatches.Patch(color=colours[l], label=l) for l in labels]
            ax.legend(handles=patches, loc=legend, framealpha=1.0)

    if grid:
        ax.grid(linestyle='--')

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

def boxplot(ax, data, colors, data_labels):
    bp = ax.boxplot( data, notch=False, labels=['']*len(data_labels), patch_artist=True )

    for b, c in zip(bp['medians'], colors):
        b.set(  color='w', linewidth=3)

    for b, c in zip(bp['boxes'], colors):
        b.set( color = c )

    for b, c in zip(bp['fliers'], colors):
        b.set( marker='+',  markeredgecolor=c, color = c, markersize=5 )


    for attr in ['caps', 'whiskers']:
        boxplots = np.array(bp[attr])
        boxplots = boxplots.reshape( (len(data_labels), 2) )
        
        for t,c in zip(boxplots, colors):
            for b in t:
                b.set( color=c, linewidth=2)
    return bp['boxes'], data_labels

def plot_mean_and_CI(ax, mean, lb, ub, color_mean, color_shading, label):
    lb[ lb < 0 ] = 0.0
    ax.fill_between(range(mean.shape[0]), ub, lb, color=color_shading, alpha=.35, label=label)
    line, = ax.plot(mean, color_mean)
    return line

def multi_line_with_CI(ax, data, colors, data_labels):
    lines = []
    for d, c, l in zip(data, colors, data_labels):
        mean = np.mean(d, axis = 0)
        std = np.std(d, axis = 0)
        std_err = std / np.sqrt(d.shape[0])
        std_err *= 1.96
        line = plot_mean_and_CI( ax, mean, mean-std_err, mean+std_err, c, c, l)
        lines.append(line)
    return lines, data_labels

def multi_line(ax, data, colors, data_labels):
    lines = []
    for d, c, l in zip(data, colors, data_labels):
        line, = ax.plot(d, color=c)
        lines.append(line)
    return lines, data_labels

def scatter(ax, x, y, colors, data_labels, scatter_size=100):
    lines = []
    ax.scatter(x, y, color=colors, s=scatter_size, alpha=0.5)
    return lines, data_labels

def get_cmap(n, name='brg'):
    return plt.cm.get_cmap(name, n)

def multi_histogram(ax, data, colors, data_labels, dmin, dmax, nbins):
    n = len(data)
    bwidth = 1.0/n
    bars = []
    for d, c, i in zip(data, colors, range(n)):
        hist, bin_edges = np.histogram(d, bins=nbins, range=[dmin, dmax])
        x = np.arange(len(hist))+(bwidth*i)
        bar = ax.bar(x, hist, width=bwidth, align='edge', color=c ) 
        bars.append(bar)

    bin_edges = bin_edges.astype(int)
    ax.set_xticks(np.arange(len(bin_edges)))
    ax.set_xticklabels([ '' if i%2==1 else str(be) for be,i in zip(bin_edges,range(len(bin_edges)))  ])

    return bars, data_labels

def save_graph(f, fname, dpi, h, w):
    f = plt.gcf()
    f.set_size_inches(w, h, forward=True)
    f.savefig(fname, dpi=dpi, bbox_inches='tight')

