# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:58:13 2017

@author: jimmybow
"""

import pandas as pd
import numpy as np 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event, State
import flask
from flask import Flask
from dfply import *
import json
import os
import time
import math
import colorlover as cl
import visdcc
import matplotlib
matplotlib.use('Agg')    ### heroku 無安裝 Tkinter  因此改成無圖片互動模式
import matplotlib.pyplot as plt


### 匯入資料
df_r = pd.read_csv('df_r.csv')
aim = pd.read_csv('aim.csv').x
freq = pd.read_csv('freq.csv',  header = None)[0]
g = pd.read_csv('g.csv', header = None)[0]
result = pd.read_csv('result.csv')
timetable = pd.read_csv('timetable.csv')

### 靜態文件位置        正式佈署時 file = __file__
file = __file__
STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(file)), 'static')

### 圖片
plt.figure(figsize = (7, 3))
ps = plt.subplot(111)
ps.set_xticks([])
ps.set_yticks([])
ps.spines['right'].set_color('none')
ps.spines['top'].set_color('none')
ps.spines['bottom'].set_color('none')
ps.spines['left'].set_color('none')
plt.xlim((1, 2))
l0, = plt.plot(0, 0, 'o', color = '#c75181', markersize=30, label='color : Influence')
l1, = plt.plot(0, 0, color='blue', linewidth = 5.0, label='color : Freq')
l2, = plt.plot(0, 0, color='black', linewidth = 10.0, label='width : Freq')
ld = plt.legend(loc='center',prop={'size':40})
ld.get_frame().set_edgecolor('black')
plt.savefig(os.path.join(STATIC_PATH, 'legend.png'))

### 時間軸參數
mint = time.strptime(df_r.time[0], "%Y-%m-%d %H:%M:%S")
maxt = time.strptime(df_r.time[len(df_r)-1], "%Y-%m-%d %H:%M:%S")
labelt = {}
for i in range(int(time.mktime(mint)),int(time.mktime(maxt)+1)):
    if (i-time.mktime(mint))% 3600==0:
        labelt.update( {i: time.strftime("%H h", time.localtime(i)) } )
        
### 起始位置
li_x = [200*math.cos(math.pi*0.2*i) for i in range(len(g))]
li_y = [200*math.sin(math.pi*0.2*i) for i in range(len(g))]

### 顏色
cg = cl.interp(['rgb(221, 212, 216)', 'rgb(142, 9, 57)'], math.ceil((len(g)+3)/10)*10) # 點的排名所對到的顏色
cc = cl.interp(['rgb(171, 176, 176)', 'rgb(2, 57, 167)'], 200) # 邊所對到的顏色
cc[0] = "#FFFFFF"

### table 
def generate_table(dataframe, max_rows = 100):
    return html.Table(
        # Header
        [html.Tr([html.Th(col,
            style = {'border-bottom-width': '2px',                             
                     'border-bottom-style': 'solid',
                     'border-bottom-color': 'rgb(221, 221, 221)',
                     'padding': '5px 12px'}) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([html.Td(dataframe.iloc[i][col], 
            style = {'border-top-width': '1px',                             
                     'border-top-style': 'solid',
                     'border-top-color': 'rgb(221, 221, 221)',
                     'padding': '5px 12px'}) for col in dataframe.columns])
            for i in range(min(len(dataframe), max_rows))],
        style = {'border-collapse':'collapse'}            
    )
    
### 網絡圖的 data
data = {}
data['nodes'] = [{'id' : g[i], 
                  'label' : g[i],
                  'shape' : 'dot',
                  'size' : 25,
                  'font.size' : 6,
                  'font.color' : "black",
                  'title' : g[i],          
                  'color': cg[0] }  for i in range(len(g))] 
data['edges'] = []

### 全域
glo = {}
glo['time-slider'] = []  
###

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(name = __name__, server = server)
app.config.supress_callback_exceptions = True

app.layout = html.Div([
      html.Div(id = 'glo', style = {'display':'none'}),  
      html.Br(),
      html.B('圖形展示', style={'font-size':20, 'display':'inline-block', 'padding': '0px 0px 0px 20px', 'width' : '15%'} ),
      html.B(style={'display':'inline-block', 'padding': '0px 0px 0px 20px', 'width' : '15%'} ),      
      html.B('流體力學', style={'font-size':20, 'display':'inline-block', 'padding': '0px 0px 0px 20px', 'width' : '15%'} ), html.Br(),    
      html.Div([dcc.Dropdown(id = 'choose-time',
                             options=[{'label': '部分時間', 'value': 'parttime'},
                                      {'label': '所有時間', 'value': 'alltime'} ],     
                             value = 'parttime'   )],
                style={'width': '15%', 'padding': '20px 0px 0px 20px', 'display':'inline-block'}   ),
      html.Div([dcc.Dropdown(id = 'choose-mode',
                             options=[{'label': '可能影響的事件', 'value': 'impact'},
                                      {'label': '可能的異常因子', 'value': 'factor'} ],     
                             value = 'impact'   )],
                style={'width': '15%', 'padding': '20px 0px 0px 20px', 'display':'inline-block'}   ),
      html.Div([dcc.Dropdown(id = 'choose-phy',
                             options=[{'label': '關', 'value': 'off'},
                                      {'label': '開', 'value': 'on'} ],     
                             value = 'off'   )],
                style={'width': '15%', 'padding': '20px 0px 0px 20px', 'display':'inline-block'}   ), html.Br(), html.Br(),                         
      html.B(id = 'time-value', style={'font-size':18, 'padding': '20px 0px 0px 30px'}),                        
      html.Div(dcc.RangeSlider(id = 'time-slider',                        
                                count = 1,
                                min = time.mktime(mint),
                                max = time.mktime(maxt),
                                step = 400,
                                value = [time.mktime(mint), time.mktime(mint) + 7000],
                                pushable = 3600,         
                                marks = labelt                                
                                ),
               style={'padding': '20px 20px 70px 20px', 'width': '65%'}  ),                
      visdcc.Network(id = 'net', 
                     data =  { 'nodes' : [{'id':g[i], 'x':li_x[i], 'y':li_y[i] }  for i in range(len(g))],
                               'edges' : [] } ,
                     selection = {'nodes':[], 'edges':[]},
                     style = {'display':'inline-block', 'height':'600px', 'width':'65%', 'padding':'20px', 'vertical-align':'top'}),
      html.Div([
          html.Img(src='/static/legend.png', style={'width': '300px'}), html.Br(), 
          html.B('可能影響的事件：', 
                 style={'font-size':20, 'display':'inline-block', 'padding':'20px', 'margin-right':'15px', 'width': '170px'} ),
          html.B('可能的異常因子：', 
                 style={'font-size':20, 'display':'inline-block', 'padding':'20px', 'width': '170px'} ), html.Br(),               
          html.Div(id = 'impact',
                   style = {'background-color':'rgb(255, 239, 225)', 'padding':'20px', 'width': '170px',
                            'vertical-align':'top', 'margin-right':'15px', 'display':'inline-block'}),
          html.Div(id = 'factor',
                   style = {'background-color':'rgb(240, 255, 225)', 'padding':'20px', 'width': '170px',
                            'vertical-align':'top', 'display':'inline-block'}),        
      ], style = {'display':'inline-block', 'text-align': 'center'})             
])

### 靜態文本位置設定
@app.server.route('/static/<resource>')
def serve_static(resource):
    return flask.send_from_directory(STATIC_PATH, resource)    
    
@app.callback(
    Output('net', 'options'),
    [Input('choose-phy', 'value')])
def myfun(x): 
    opt = {'physics' : {'enabled': False}}
    if x == 'on':
        opt = {'physics' : {'enabled': True}}
    return opt   
      
@app.callback(
    Output('time-slider', 'disabled'),
    [Input('choose-time', 'value')])
def myfun(x): 
    boo = False
    if x == 'alltime':
        boo = True
    return boo

@app.callback(
    Output('time-slider', 'value'),
    [Input('choose-time', 'value')])
def myfun(x): 
    ss = [time.mktime(mint), time.mktime(mint) + 7000]
    if x == 'alltime':
        ss = [time.mktime(mint), time.mktime(maxt)]
    return ss

@app.callback(
    Output('time-value', 'children'),
    [Input('time-slider', 'value')])
def myfun(x): 
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x[0])) + ' 至 ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x[1])) 

@app.callback(
    Output('glo', 'children'),
    [Input('time-slider', 'value'),
     Input('choose-mode', 'value'),
     Input('net', 'selection')],
    state = [State('glo', 'children')])     
def myfun(x, mode, select, ggg):    
    ddf = 'null'
    if x != glo['time-slider']:    
        glo['time-slider'] = x
        s_value = x[0]
        e_value = x[1]
        if e_value > time.mktime(maxt): e_value = time.mktime(maxt)  
        
        s_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s_value)) 
        e_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e_value)) 
        
        s = (s_str >= timetable.start).sum() - 1
        e = (e_str > timetable.end).sum()    
        ww = [i for i in range(s, e + 1)]
        kk = result.lhs.isin(timetable.time[ww])
        glo['kk'] = kk
        glo['node_color'] = [cg[0]  for i in range(len(g))]     
        if kk.sum() > 0 :
            st = result.rhs[kk].str.split('_')
            fr = []
            to = []
            for i in range(len(st)):
               for j in range(1, len(st.iloc[i])):
                 fr.append(st.iloc[i][j-1])
                 to.append(st.iloc[i][j])
            
            df_E = pd.concat([pd.Series(fr, name = 'from'), pd.Series(to, name = 'to')], axis = 1)
            
            ### 點的顏色設置 
            hg = (df_E >> groupby(X['from']) >> summarize(Freq = n(X.to)) >> ungroup()
                       >> right_join(pd.DataFrame(pd.Series(g, name = 'from')))  >> arrange(X['from'])    )        
            nh = hg.Freq.isnull().sum()
            ra = hg.Freq.rank(na_option='top')
            ra[hg.Freq.isnull()] = 1
            ra = list(map(lambda x: math.floor(x), ra))   # 點的顏色依據重要性的排名   
            glo['node_color'] = [cg[ra[i]]  for i in range(len(g))]         
    
            ### 邊的設置 
            ww2 = (df_r.time <= e_str) & (df_r.time >= s_str) & (~aim.isnull())
            df_E2 = df_E >> groupby(X['from'], X.to) >> summarize(Freq = n(X.to)) >> ungroup() >> mutate(id = X['from'] + '--' + X.to) 
               
            s = '--'.join(aim[ww2].tolist()).split('--')
            sf = '--'.join(freq[ww2].tolist()).split('--')
            Len = list(map(lambda x: len(x),freq[ww2].str.split('--')))
            event_name = ''.join(list(map(lambda x, y: x*(y + '--'), Len, df_r.event[ww2] ))).split('--')
            event_name.remove('')
            ddf = (pd.DataFrame.from_dict({'from':event_name, 'to':s, 'f':sf}).astype({'f':int})
                   >> groupby(X['from'], X.to) >> summarize(FFreq = X.f.sum()) >> ungroup() 
                   >> right_join(df_E2) ).dropna()
            
            ddf['width'] = [max(min(ddf.FFreq.iloc[i]*0.0001, 5), 1) for i in range(len(ddf))]
            ddf['selectionWidth'] = 0  
    
    nc = list(glo['node_color'])
    # 選擇互動            
    if type(ddf) == str: ddf = pd.DataFrame(json.loads(json.loads(ggg)['ddf']))
    nn = len(ddf)
    ddf['color'] = [ cc[min(math.ceil(ddf.FFreq.iloc[i]*0.01), 199)]    for i in range(nn)]
    ddf['hidden'] = False   
    ddf['arrows'] = [{'to':{'enabled': False} }  for i in range(nn)]
    ddf['smooth'] = [{ 'enabled' : False } for i in range(nn)]
    idd = []
    
    if len(select['nodes']) > 0 : 
        gid = list(ddf.id)
        sel_aim = select['nodes'][0]        
        if mode == 'impact':       
            ww3 = result.rhs[glo['kk']].str.contains(sel_aim + '_')
            st = result.rhs[glo['kk']][ww3].str.replace('[a-zA-Z_]*_' + sel_aim , sel_aim).str.split('_')
            cgc = '#FF8040'
            for i in range(len(st)):
                nni = len(st.iloc[i])
                for j in range(1, nni):
                    v = st.iloc[i][j-1] + '--' + st.iloc[i][j]
                    if not v in gid : break     # 順方向接鍊子 鍊子斷了就停止
                    else : idd.append(v)   
        else :
            ww3 = result.rhs[glo['kk']].str.contains('_' + sel_aim)
            st = result.rhs[glo['kk']][ww3].str.replace(sel_aim + '_[a-zA-Z_]*', sel_aim).str.split('_')
            cgc = '#00DB00'
            for i in range(len(st)):
                nni = len(st.iloc[i])
                for j in range(nni-1, 0, -1):
                    v = st.iloc[i][j-1] + '--' + st.iloc[i][j]
                    if not v in gid : break      # 反方向接鍊子 鍊子斷了就停止
                    else : idd.append(v)   
           
        ww = ddf.id.isin(idd)
        ddf['color'][ww] = cgc   # 邊的顏色
        ddf['hidden'][~ww] = True
        ddf['arrows'][ww] = [{'to':{'enabled': True} }  for i in range(ww.sum())] 
           
        # 點的顏色   glo['node_color']
        nid = ddf.id[ww].tolist()
        non = pd.Series('--'.join(nid).split('--')).unique()
        for i in range(len(g)):    
            if data['nodes'][i]['id'] in non : nc[i]= cgc     
            if data['nodes'][i]['id'] == select['nodes'][0] : nc[i]= 'red'      
        
    data['edges'] = ddf.to_dict('records')    
    for i in range(nn):
        if len(select['nodes']) == 0 and len(select['edges']) > 0 and select['edges'][0] == data['edges'][i]['id'] :
            data['edges'][i]['color'] = 'red'  
            
    # 點的顏色設置
    for i in range(len(g)): data['nodes'][i]['color'] = nc[i] 
    
    return json.dumps({'data': data, 'idd': idd, 'ddf': ddf.to_json() })

@app.callback(
    Output('net', 'data'),
    [Input('glo', 'children')])
def myfun(d):
    return json.loads(d)['data']

@app.callback(
    Output('impact', 'children'),
    [Input('glo', 'children')],
    state = [State('net', 'selection'),
             State('choose-mode', 'value')])
def myfun(d, sel, mode):
    idd = json.loads(d)['idd']
    table = pd.DataFrame(columns = ['Impack', 'Rank'])
    if mode == 'impact' and len(sel['nodes']) > 0 and len(idd) > 0 :
        table = pd.Series(idd).str.replace('[a-zA-Z]*--','').value_counts().rank(ascending = False).astype(int).reset_index()
        table.columns = ['Impack', 'Rank']        
    return generate_table(table)

@app.callback(
    Output('factor', 'children'),
    [Input('glo', 'children')],
    state = [State('net', 'selection'),
             State('choose-mode', 'value')])
def myfun(d, sel, mode):
    idd = json.loads(d)['idd']
    table = pd.DataFrame(columns = ['Factor', 'Rank'])
    if mode == 'factor' and len(sel['nodes']) > 0 and len(idd) > 0 :
        table = pd.Series(idd).str.replace('--[a-zA-Z]*','').value_counts().rank(ascending = False).astype(int).reset_index()
        table.columns = ['Factor', 'Rank']        
    return generate_table(table)
