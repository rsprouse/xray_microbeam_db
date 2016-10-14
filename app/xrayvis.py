#!/usr/bin/env python

import os, sys
import fnmatch
import numpy as np
import scipy.io.wavfile
import scipy.signal
import pyaudio

import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Span, BoxAnnotation
from bokeh.models.tools import \
     CrosshairTool, BoxZoomTool, BoxSelectTool, HoverTool, \
     PanTool, ResetTool, SaveTool, TapTool, WheelZoomTool
from bokeh.models.widgets import Div, Slider, TextInput, PreText, Select, Button
from bokeh.plotting import figure, output_file, output_notebook, show
from bokeh.document import without_document_lock
from tornado import gen

def play_all():
    audata = orig_au.astype(np.int16).tostring()
    stream.write(audata)

def play_sel():
    sel = gp.select_one(dict(tags=['cursel']))
    samp1 = np.int(sel.left * orig_rate)
    samp2 = np.int(sel.right * orig_rate)
    audata = orig_au[samp1:samp2].astype(np.int16).tostring()
    stream.write(audata)

def get_filenames():
    '''Walk datadir and get all .wav filenamess.'''
    files = []
    for root, dirnames, fnames in os.walk(datadir):
        for fname in fnmatch.filter(fnames, '*.wav'):
            files.append(os.path.join(root, fname))
    return files

def load_file(attrname, old, wav):
    global au, orig_au, rate, orig_rate, timepts, df, stream
    (orig_rate, au) = scipy.io.wavfile.read(wav)
    orig_au = au.copy()
    if stream is None:
        # Set up the audio stream for playback.
        pya = pyaudio.PyAudio()
        stream = pya.open(
                    format = pyaudio.paInt16,
                    channels = 1,
                    rate = np.int(orig_rate),
                    output = True)
    decim_factor = 16
    au = scipy.signal.decimate(au, decim_factor)
    rate = orig_rate / decim_factor
    timepts = np.arange(0, len(au)) / rate
    source.data['x'] = timepts
    source.data['au'] = au
    x_range.update(end=timepts[-1])

    # Now load the tongue data
    tngfile = os.path.splitext(wav)[0] + '.txy'
    palfile = os.path.join(os.path.dirname(wav), 'PAL.DAT')
    phafile = os.path.join(os.path.dirname(wav), 'PHA.DAT')
    df = pd.read_csv(
            tngfile,
            sep='\t',
            names=[
                'sec', 'ULx', 'ULy', 'LLx', 'LLy', 'T1x', 'T1y', 'T2x', 'T2y',
                'T3x', 'T3y', 'T4x', 'T4y', 'MIx', 'MIy', 'MMx', 'MMy'
            ]
        )
    df['sec'] = df['sec'] / 10e5
    df = df.set_index(['sec'])
    paldf = pd.read_csv(palfile, sep='\s+', header=None, names=['x', 'y'])
    palsource.data = {'x': paldf['x'], 'y': paldf['y']}
    phadf = pd.read_csv(phafile, sep='\s+', header=None, names=['x', 'y'])
    phasource.data = {'x': phadf['x'], 'y': phadf['y']}

    
def make_plot():
    '''Make the plot figures.'''
    ts = []
    ts.append(figure(
            width=width, height=height,
            title="Audio", y_axis_label=None,
            x_range=(0,30),
            tools=tools[0], webgl=True
        )
    )
    ts[0].line('x', 'au', source=source, tags=['update_ts'])
    ts[0].circle('x', 'au', source=source, size=0.1, tags=['update_ts'])
    curcur = Span(location=0, dimension='height', tags=['curtime'])
    cursel = BoxAnnotation(left=0, right=0, fill_alpha=0.1, fill_color='blue', tags=['cursel'])
    ts[0].add_layout(curcur)
    ts[0].add_layout(cursel)
    ts.append(figure(
            width=500, height=300,
            title='Static trace',
#            x_range=(-100000,20000), y_range=(-30000,30000),
            tools=tools[1], webgl=True,
            tags=['static_fig']
        )
    )
    ts[1].circle('x', 'y', source=tngsource, size=3, color=tngcolor, tags=['update_xray'])
    ts[1].circle('x', 'y', source=othsource, size=3, color=othcolor, tags=['update_xray'])
    ts[1].line('x', 'y', source=tngsource, color=tngcolor, tags=['update_xray'])
    ts[1].line('x', 'y', source=palsource, color='black')
    ts[1].line('x', 'y', source=phasource, color='black')
    ts.append(figure(
            width=500, height=300,
            title='Trajectories',
#            x_range=(-100000,20000), y_range=(-30000,30000),
            tools=tools[2], webgl=True,
            tags=['trajectory_fig']
        )
    )
    ts[2].line('T1x', 'T1y', source=timesource, color=tngcolor, tags=['update_xray'])
    ts[2].line('T2x', 'T2y', source=timesource, color=tngcolor, tags=['update_xray'])
    ts[2].line('T3x', 'T3y', source=timesource, color=tngcolor, tags=['update_xray'])
    ts[2].line('T4x', 'T4y', source=timesource, color=tngcolor, tags=['update_xray'])
    ts[2].line('ULx', 'ULy', source=timesource, color=othcolor, tags=['update_xray'])
    ts[2].line('LLx', 'LLy', source=timesource, color=othcolor, tags=['update_xray'])
    ts[2].line('MIx', 'MIy', source=timesource, color=othcolor, tags=['update_xray'])
    ts[2].line('MMx', 'MMy', source=timesource, color=othcolor, tags=['update_xray'])
    ts[2].circle('x', 'y', source=lasttngtimesource, color=tngcolor, tags=['update_xray'])
    ts[2].circle('x', 'y', source=lastothtimesource, color=othcolor, tags=['update_xray'])
    ts[2].line('x', 'y', source=palsource, color='black')
    ts[2].line('x', 'y', source=phasource, color='black')
    gp = gridplot([[ts[0]], [ts[1], ts[2]]])
    return (gp, ts[0])

def update_data_0d(sec):
    '''Update the static trace at selected time.'''
    tidx = df.index.get_loc(sec, method='nearest')
    row = df.iloc[tidx]
    tngsource.data = {
        'x': [row.T1x, row.T2x, row.T3x, row.T4x],
        'y': [row.T1y, row.T2y, row.T3y, row.T4y]
    }
    othsource.data = {
        'x': [row.ULx, row.LLx, row.MIx, row.MMx],
        'y': [row.ULy, row.LLy, row.MIy, row.MMy]
    }

def update_data_1d(t1, t2):
    '''Update the trajectories in selected range.'''
    t1idx = df.index.get_loc(t1, method='nearest')
    t2idx = df.index.get_loc(t2, method='nearest')
    seldf = df.iloc[t1idx:t2idx]
    timesource.data = seldf.to_dict('list')
    lastrow = seldf.iloc[-1]
    lasttngtimesource.data = {
        'x': [lastrow.T1x, lastrow.T2x, lastrow.T3x, lastrow.T4x],
        'y': [lastrow.T1y, lastrow.T2y, lastrow.T3y, lastrow.T4y]
    }
    lastothtimesource.data = {
        'x': [lastrow.ULx, lastrow.LLx, lastrow.MIx, lastrow.MMx],
        'y': [lastrow.ULy, lastrow.LLy, lastrow.MIy, lastrow.MMy]
    }

def selection_change(attr, old, new):
    sys.stderr.write("*****selection_change***********\n")
    if len(new['1d']['indices']) > 1:
        ind = new['1d']['indices']
        x1sel = np.min(ind)
        x2sel = np.max(ind)
        t1sel = x1sel / rate
        t2sel = x2sel / rate
        update_data_1d(t1sel, t2sel)
        gp.select_one(dict(tags=['trajectory_fig'])).title.text = \
            'Trajectories ({:0.4f}-{:0.4f})'.format(t1sel, t2sel)
        sel = gp.select_one(dict(tags=['cursel']))
        sel.left = t1sel
        sel.right = t2sel
    elif len(new['0d']['indices']) > 0:
        curtime = new['0d']['indices'][0] / rate
        gp.select_one(dict(tags=['static_fig'])).title.text = \
            'Static trace ({:0.4f})'.format(curtime)
        gp.select_one(dict(tags=['curtime'])).location = curtime
        update_data_0d(curtime)
    else:
        print('no indices')


# Filename selector
datadir = os.path.join(os.path.dirname(__file__), 'data')
fsel = Select(options=['Select a file'] + get_filenames())
play_all_button = Button(label='All', button_type='success', width=60)
play_sel_button = Button(label='Sel', button_type='success', width=60)
tngcolor = 'DarkRed'
othcolor = 'Indigo'
#msgdiv = Div(text='', width=400, height=50)


step = None
rate = orig_rate = None
df = None
stream = None
au = orig_au = timepts = []
pelx = pely = othx = othy = []
width = 1000
height = 200
cutoff = 50
order = 3
source = ColumnDataSource(
    data=dict(
        x=timepts,
        au=au
    )
)
tngsource = ColumnDataSource(
    data=dict(
        x=pelx,
        y=pely
    )
)
othsource = ColumnDataSource(
    data=dict(
        x=othx,
        y=othy
    )
)
timesource = ColumnDataSource(
    pd.DataFrame(
        {
            'T1x': [], 'T1y': [], 'T2x': [], 'T2y': [],
            'T3x': [], 'T3y': [], 'T4x': [], 'T4y': [],
            'ULx': [], 'ULy': [], 'LLx': [], 'LLy': [],
            'MIx': [], 'MIy': [], 'MMx': [], 'MMy': []
        }
    )
)
lasttngtimesource = ColumnDataSource(
    data=dict(
        x=[],
        y=[]
    )
)
lastothtimesource = ColumnDataSource(
    data=dict(
        x=[],
        y=[]
    )
)

palsource = ColumnDataSource(pd.DataFrame({'x': [], 'y': []}))
phasource = ColumnDataSource(pd.DataFrame({'x': [], 'y': []}))

# Create the tools for the toolbar
ts_cnt = np.arange(3)
#cross = [CrosshairTool(dimensions=['height']) for n in ts_cnt]
hover = [
    HoverTool(tooltips=[('time', '$x')]),
    HoverTool(tooltips=[('x', '$x'), ('y', '$y')]),
    HoverTool(tooltips=[('x', '$x'), ('y', '$y')])
]
xzoom = [BoxZoomTool(dimensions=['width']) for n in ts_cnt]
xwzoom = [WheelZoomTool(dimensions=['width']) for n in ts_cnt]
xsel = [BoxSelectTool(dimensions=['width']) for n in ts_cnt]
xtsel = [TapTool() for n in ts_cnt]
xpan = [PanTool() for n in ts_cnt]
save = [SaveTool() for n in ts_cnt]
reset = [ResetTool() for n in ts_cnt]
tools = [
    [
        #cross[n],
        hover[n], xpan[n], xzoom[n], xwzoom[n],
        xsel[n], xtsel[n], save[n], reset[n]
    ]
    for n in ts_cnt
]
source.on_change('selected', selection_change)
fsel.on_change('value', load_file)
play_all_button.on_click(play_all)
play_sel_button.on_click(play_sel)

curdoc().add_root(row(play_all_button, play_sel_button, fsel)) #, msgdiv))
gp, ch0 = make_plot()
x_range = ch0.x_range
curdoc().add_root(gp)
