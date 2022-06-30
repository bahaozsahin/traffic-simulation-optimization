import os, argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mp
from matplotlib.colors import LinearSegmentedColormap

from graph_globals import global_params
from graphs import graph, boxplot, multi_line, multi_line_with_CI, get_cmap, scatter, save_graph
from utils.pickle_helper import load_data
from utils.helpers import check_and_make_dir

def main():
    global_params()
    colours = ['blue', 'red', 'orange', 'y', 'm']
    labels = {'ddpg':'DDPG', 'dqn':'DQN', 'maxpressure':'Max-pressure', 'websters':'Webster\'s', 'fixed':'Fixed'}
    if len(colours) != len(labels):
        assert 0, 'Error: the number of colours '+str(len(colours))+' does not equal the number of labels'+str(len(labels))

    colours = { l:c  for c, l in zip(colours, labels)}

    args = parse_cl_args()
    check_and_make_dir(args.save_dir)

    if args.type == 'moe':
        fp = 'metrics/'
        graph_travel_time(labels, colours, fp, args.save_dir)
        metrics = ['queue', 'delay']
        graph_individual_intersections(labels, colours, fp, metrics, args.save_dir)
    elif args.type == 'hp': 
        fp = 'hp/'
        graph_hyper_params(labels, colours, fp, args.save_dir)
    else:
        assert 0, print('Error, supplied graph type argument '+str(args.type)+' does not exist')

def parse_cl_args():
    parser = argparse.ArgumentParser()

    ##sumo params
    parser.add_argument("-type", type=str, default='moe', dest='type', help='Data to be graphed, default: moe, options: moe, hp')
    parser.add_argument("-save_dir", type=str, default='figures/', dest='save_dir', help='Directory to save figures, default: figures/')
    args = parser.parse_args()
    return args

def graph_hyper_params(labels, colours, fp, save_dir):
    tsc = os.listdir(fp)
    tsc_hp = {}
                                                                                                                   
    for t in tsc:
        tsc_fp = fp+t+'/'
        data = [ load_data(tsc_fp+f) for f in os.listdir(tsc_fp)]                                                 
        tsc_hp[t] = np.stack([ [np.mean(d), np.std(d)] for d in data]).T

    n = len(tsc)
    if n == 1:
        f, axes = plt.subplots()
        axes = [axes]
    else:
        nrows = 2 
        ncols = int(n/nrows) if n%nrows == 0 else int((n+1)/nrows)
        f, axes = plt.subplots(nrows=nrows,ncols=ncols)
        axes = axes.flat
    
    if n%nrows != 0:
        f.delaxes(axes[-1])

    XTITLE = 'Mean\nTravel Time '+r"$(s)$"
    YTITLE = ('Standard\nDeviation\nTravel Time '+r"$(s)$", 80)

    for ax, t, i in zip(axes, tsc, range(len(tsc))):
        mean_data = tsc_hp[t][0] 
        std_data = tsc_hp[t][1]
        data = sorted([ (m+s, m, s) for m,s in zip(mean_data, std_data) ], key = lambda x:x[0] )
        data = np.stack([ [d[1], d[2]] for d in data]).T
        mean_data = data[0]
        std_data = data[1]


        rg_colours = mp.cm.brg(np.linspace(1.0, 0.5, len(mean_data)))                          
        if i%ncols == 0 and i >= len(tsc)/2:
            xtitle = XTITLE 
            ytitle = YTITLE 
        elif i%ncols == 0:
            xtitle = ''
            ytitle = YTITLE
        elif i >= len(tsc)/2:
            xtitle = XTITLE
            ytitle = ''
        else:
            xtitle = ''
            ytitle = ''
        graph( ax, mean_data, scatter( ax, mean_data, std_data, rg_colours, ['']*len(mean_data)), 
                                  xtitle=xtitle,                                 
                                  ytitle_pad = ytitle,       
                                  title=str(labels[t]),        
                                  xlim = [0.0, max(mean_data)*1.05],                          
                                  ylim= [0.0, max(std_data)*1.05],                            
                                  grid=True)                                                  

    cax = f.add_axes([0.915, 0.1, 0.05, 0.85])
    cmap = mp.cm.brg
    cm = LinearSegmentedColormap.from_list('rg', rg_colours, N=rg_colours.shape[0])
    norm = mp.colors.Normalize(vmin=0.5, vmax=1.0)
    cb = mp.colorbar.ColorbarBase(cax, cmap=cm,
                                norm=norm,
                                orientation='vertical')

    cb_labels = [ l._text for l in cb.ax.get_yticklabels()]
    cb_labels[0] = 'Best'
    cb_labels[-1] = 'Worst'
    cb.ax.set_yticklabels(cb_labels)

    f.suptitle('Hyperparameter Performance')                                                            
    save_graph(f, save_dir+'tsc_hp.pdf', 600, 14, 24.9)
    plt.show()                                                                            

    data_order = sorted(tsc_hp.keys())
    mean_data, std_data, colors, tsc_labels = [], [], [], []
    for d in data_order:
        n = len(tsc_hp[d][0])
        mean_data.extend(tsc_hp[d][0])
        std_data.extend(tsc_hp[d][1])
        tsc_labels.extend(labels[d])
        colors.extend( [colours[d]]*n )
    f, ax = plt.subplots(1,1)
    graph( ax, mean_data, scatter( ax, mean_data, std_data, colors, ['']*len(mean_data)),
                              xtitle=XTITLE,
                              ytitle_pad = YTITLE,
                              title='Traffic Signal Control\nHyperparameter Comparison',
                              xlim = [0.0, 200.0],
                              ylim= [0.0, 200.0],
                              grid=True)


    patches = []
    for d in data_order:
        c = colours[d]
        patches.append( mpatches.Patch(color=c, label=labels[d]) )
    plt.legend(handles=patches, framealpha=1.0)
    save_graph(f, save_dir+'hp.pdf', 600, 14, 24.9)
    plt.show()

def graph_travel_time(labels, colours, fp, save_dir):
    data = get_data(fp, 'traveltime', get_folder_data)                               
    data_order = sorted(data.keys())                                                 
    data = [ data[d] for d in data_order]                                           
    labels = [ labels[d] for d in data_order]                                       
    c = [ colours[d] for d in data_order]                                       
    f, ax = plt.subplots(1,1)                                                        
    t = 'Travel Time '+r"$(s)$"+'\n('+r"$\mu,\sigma,$"+'median'+r"$)$"
    graph( ax, data, boxplot( ax, data, c, labels),                            
                              xtitle='Traffic Signal Controller',                    
                              ytitle_pad = (t, 60),                      
                              title='Traffic Signal Controller\nTravel Time ',     
                              legend=(0.82, 0.72),                                   
                              grid=True)                                             

    for i, d in enumerate(data_order):
        text = '('+str(int(np.mean(data[i])))+', '+str(int( np.std(data[i]) ) )+', '+str(int( np.median(data[i]) ) )+r"$)$"
        ax.text(i+1.1, 300, text, color= c[i])

    save_graph(f, save_dir+'travel_time.pdf', 600, 14, 24.9)
    plt.show()                                                                       

def graph_conf_interval(labels, colours, fp, metric):
    data = get_data(fp, metric, get_metric_data)                         
    data_order = sorted(data.keys())                                     
    data = [ data[d]  for d in data_order]                               
    labels = [ labels[d]  for d in data_order]                           
    f, ax = plt.subplots(1,1)                                            
    metric_title = metric.capitalize()                                   
    graph( ax, data, multi_line_with_CI( ax, data, colours, labels),     
           xtitle='Time (s)',                                            
           ytitle_pad = (metric_title, 60),                              
           title=metric_title+' by\nTraffic Signal Controller',          
           legend=(0.72, 0.72),                                          
           grid=True)                                                    
    plt.show()                                                           

def get_data(fp, metric, read_data_func):
    tsc = os.listdir(fp)
    tsc_data = { t:read_data_func(fp+t+'/'+metric) for t in tsc}
    return tsc_data

def get_metric_data(fp):
    if not os.path.exists(fp):
        assert 0, 'Supplied path '+str(fp)+' does not exist.'

    tsc_data = {tsc_id:sorted(os.listdir(fp+'/'+tsc_id)) 
                    for tsc_id in os.listdir(fp)}
    sim_runs_data = []

    k = list(tsc_data.keys())[0]
    while len(tsc_data[k]) > 0:
        same_run_data = [ fp+'/'+tsc_id+'/'+tsc_data[tsc_id].pop(0) 
                          for tsc_id in tsc_data ]
        same_run_data = [ load_data(f) for f in same_run_data ]
        sim_runs_data.append( np.sum(same_run_data, axis=0) )

    return np.stack(sim_runs_data)

def get_folder_data(fp):
    if not os.path.exists(fp):
        assert 0, 'Supplied path '+str(fp)+' does not exist.'
    data = []
    for f in os.listdir(fp):
        data.extend(load_data(fp+'/'+f))
    return np.array(data)

def stack_folder_files(fp):
    data = [ load_data(fp+f) for f in os.listdir(fp)]
    return np.stack(data)

def graph_individual_intersections(labels, colours, fp, metrics, save_dir):
    tsc = os.listdir(fp)                          
    intersections = os.listdir(fp+tsc[0]+'/'+metrics[0]+'/')
    ncols = len(intersections)
    nrows = len(metrics)

    f, ax = plt.subplots(nrows=nrows,ncols=ncols)
    if ax.ndim == 1:
        ax = ax[...,np.newaxis]

    for m, r in zip(metrics, range(nrows)):
        data = {}
        for t in tsc:
            data[t] = {}
            for i in intersections:
                alias_p = 60
                data[t][i] = alias( stack_folder_files(fp+'/'+t+'/'+m+'/'+i+'/'), alias_p)

        xtitle = 'Time '+r" $(min)$" if r == nrows-1 else ''
        for I, c in zip(intersections, range(ncols)):
            title = I if r == 0 else ''
            if m == 'queue':
                ytitle = m.capitalize()+r" $(veh)$" if c == 0 else ''
            else:
                ytitle = m.capitalize()+r" $(s)$" if c == 0 else ''
            legend = (0.59, 0.6)
                
            data_order = sorted(data.keys())               
            alias_p = 60 
            order_data = [ data[d][I] for d in data_order]     

            order_labels = [ labels[d] for d in data_order]
            order_colors = [ colours[d] for d in data_order]

            label_c = { labels[d]:colours[d] for d in data_order}
            graph( ax[r,c], order_data, multi_line_with_CI( ax[r,c], order_data, order_colors, order_labels),
                   xtitle=xtitle,
                   ytitle_pad = (ytitle , 60),
                   title=title,
                   legend=legend,
                   colours = label_c,
                   grid=True)
            ax[r,c].set_xlim(left=0)
            ax[r,c].set_ylim(bottom=0)
    f.suptitle('Intersection Measures of Effectiveness')                                             
    save_graph(f, save_dir+'intersection_moe.pdf', 600, 14, 24.9)
    plt.show() 

def alias(data, a):
    n = data.shape[-1]
    if n % a != 0:
        a = 1

    stop = n-a
    m = int(n/a)

    alias_data = []
    for d in data:
        alias_data.append( np.array([np.sum(d[i*a:(i+1)*a]) for i in range(m) ]) )

    return np.stack(alias_data)

if __name__ == '__main__':
    main()