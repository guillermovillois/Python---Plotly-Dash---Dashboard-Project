import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import numpy as np
import time 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
targets = pd.read_csv('target.csv', index_col=0)
financials = pd.read_csv('Financials_Final.csv', index_col=0)
reportlines = pd.read_csv('reportlines.csv')
employeelevels = pd.read_csv('employeelevels.csv', index_col=0)
months= pd.read_csv('meses.csv', index_col=0)
financials = pd.merge(financials, months, on='FiscalMonthNumber')
financials = pd.merge(financials, reportlines[['ReportLine','Number','Groups']], left_on='ReportLine3', right_on='ReportLine')
HC = pd.read_csv('HC_Final.csv', index_col=0)
emptyMonths = pd.read_csv('emptyMonths.csv', index_col=0)
orglevels = pd.read_csv('orglevels.csv', index_col=0).fillna('')
colors = {
    'background': '#FFFFFF',
    'text': '#1ac000'
}
def human_format(num):
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

app.title = 'FY20 Forecast Live'

app.layout =  html.Div(style={'font-family': 'Graphik','backgroundColor': colors['background']},children=[
    html.Img(
                src='https://image.freepik.com/vector-gratis/plantilla-logo-degradado-forma-abstracta_23-2148204210.jpg',
                #https://www.accenture.com/t20180820t080654z__w__/ar-es/_acnmedia/accenture/dev/redesign/acc_logo_black_purple_rgb.png',
                style={
                    'height' : '10%',
                    'width' : '10%',
                    'float' : 'right',
                    'position' : 'relative',

                },
            ),
    html.Div([html.Div([html.H1(children='FY20 Forecast Live',
        style={
            'color': colors['text']
        }, className="six columns")]),],className="row"),
    
    html.Div([html.Div([html.Label('Org Level 3',style={'color': colors['text'], 'text-align': 'left'}, className="six columns")]),
    html.Div([html.Label('Org Level 4',style={'color': colors['text'],'text-align': 'right'}, className="six columns")])],className="row"),
    html.Div([html.Div([dcc.Dropdown(
        id='team-column',
        options=[{'label': each, 'value': each} for each in financials.Level3Name.unique()][1:],
        value=[],
        multi=True,
        style={'backgroundColor': colors['background'],
            'color': colors['text']}, className="six columns"
    )]),
    html.Div([dcc.Dropdown(
        id='team-column-level4',
        value=[],
        multi=True,
        style={'backgroundColor': colors['background'],
            'color': colors['text']
        })], className="six columns"
    )],className="row"),
    html.Div([html.Div([dcc.RadioItems(
        id='Actuals',
        options=[
            {'label': 'Actuals FY19', 'value': 'Actuals'},
            {'label': u'Actuals FY19 | Fx FY20', 'value': 'NFY'},
        ],
        value='Actuals',
        style={
            'color': colors['text']
        }
    )],style={'position' : 'relative'}, className="six columns"),
    html.Div([dcc.RadioItems(
        id='NetCost',
        options=[
            {'label': 'Gross Cost', 'value': 'GrossCost'},
            {'label': 'Net Cost', 'value': 'NetCost'},
        ],
        value='NetCost',
        style={
            'color': colors['text']
        }
    )],style={'position' : 'relative'}, className="six columns")],className="row"),


    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='SUMMARY', children=[html.Div(
            [html.Div([dcc.Graph(id='graph-with-grouptotal')]
            ,style={'width': '40%','position' : 'relative'}, className="six columns"),
            html.Div([dcc.Graph(id='graph-with-filter')], 
            style={'width': '50%','position' : 'relative'}, className="six columns"),
            ])
        ]),

        dcc.Tab(label='FINANCIALS', children=[html.Div(
            [html.Div([dcc.Graph(id='graph-with-filter1')]
    ,style={'width': '50%','position' : 'relative'}, className="six columns"),
            html.Div([dcc.Graph(id='graph-with-groups')]
    ,style={'width': '45%','position' : 'relative'}, className="six columns")
            ])]),

        dcc.Tab(label='HC', children=[html.Div([
             html.Div(
            [html.Div([dcc.Graph(id='graph-with-HC')]
    ,style={'width': '40%','position' : 'relative'}, className="six columns"),
            html.Div([dcc.Graph(id='cost-class')]
    ,style={'width': '50%','position' : 'relative'}, className="six columns")],className='row'),
            html.Div(
            [html.Div([dcc.Graph(id='trend-with-HC')],className='container', style={'maxWidth': '650px'})],className='row')])])
            ])])

@app.callback(
    Output('team-column-level4', 'options'),
    [Input('team-column', 'value')])
def select_levl4(team):
    df = orglevels[orglevels.Level3Name.isin(team)]
    return [{'label': each, 'value': each} for each in df.Level4Name.unique()]

@app.callback(
    [Output('graph-with-groups', 'figure'),
    Output('graph-with-grouptotal', 'figure'),
    Output('graph-with-filter','figure'),
    Output('graph-with-filter1', 'figure'),],
    [Input('team-column', 'value'), 
    Input('team-column-level4', 'value'),
    Input('NetCost', 'value'),
    Input('Actuals', 'value')])

def set_group_figure(team, team4, NetCost, Actuals):
    print(Actuals)
    start = time.time()

    if Actuals=='Actuals':
        financials1 = financials[(financials['Source']=='Actuals') | (financials['Source']=='Forecast')]
    else:
        financials1 = financials[(financials['Source']=='NFY') | (financials['Source']=='Forecast')]
        financials1['Source'].loc[financials1.Source=='NFY']='Actuals'
    print(financials1.head())

    if ((team==[]) & (NetCost=='NetCost')):
        df = financials1
    elif ((team==[]) & (NetCost!='NetCost')):
        df = financials1[np.in1d(financials1.GrossCostClassification,'Gross Cost')]
    elif ((len(team)>0) & (len(team4)>0) & (NetCost=='NetCost')):
        df = financials1[np.in1d(financials1.Level3Name, team) & np.in1d(financials1.Level4Name,team4)]
    elif ((len(team)>0) & (len(team4)>0)):
        df = financials1[np.in1d(financials1.Level3Name,team) & np.in1d(financials1.Level4Name,team4) & np.in1d(financials1.GrossCostClassification,'Gross Cost')]
    elif ((len(team)>0) & (NetCost=='NetCost')):
        df = financials1[np.in1d(financials1.Level3Name,team)]
        if df.shape[0]==0:
            return {'data': []},{'data': []},{'data': []},{'data': []}
    elif (len(team)>0):
        df = financials1[np.in1d(financials1.Level3Name,team) & np.in1d(financials1.GrossCostClassification,'Gross Cost')]
        if df.shape[0]==0:
            return {'data': []},{'data': []},{'data': []},{'data': []}
    no_actuals = df[np.in1d(df.Source,'Actuals')].shape[0]
    no_forecast = df[np.in1d(df.Source,'Forecast')].shape[0]
    if (no_actuals==0) or (no_forecast==0):
        source = 'Actuals' if no_actuals==0 else 'Forecast'
        df2 = df.copy()
        df2.loc[:,'Amount']=0
        df2.loc[:,'Source']=source
        df = pd.concat([df, df2])

    df = df[~(df['Source']=='NFY')]

    ##Grouping DNPS
    # Data = pd.DataFrame(
    Data = pd.pivot_table(df, columns='Source',index='Groups',values='Amount', aggfunc=sum).round().fillna(0)
    # .to_records())

    Data.loc[:,'YoY Var'] = (Data['Forecast']-Data['Actuals'])
    Groups = ['Payroll', 'Recoveries','DNP','Other']

    # Data = pd.DataFrame((Data.set_index('Groups')).loc[Groups].to_records())
    Data = Data.reindex(Groups)

    fig1 = go.Figure()
    colors = ["rgb(150,150,150)","rgb(70,0,115)","rgb(0,142,255)"]
    i = 0
    for each in Data.columns:
        fig1.add_trace(go.Bar(
            x=Groups,
            y=Data[each].tolist(),
            name=each,
            # orientation='h',
            marker_color=colors[i]
        ))

        i+=1
    
    
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig1.update_layout(title='Actual FY19 vs Forecast FY20 by RL Group',plot_bgcolor='#FFFFFF',paper_bgcolor='#FFFFFF',font ={'color': '#7500c0'},barmode='group', xaxis_tickangle=-45)

    #Grouping MD&I and others
    prod = df
    # prod.loc[prod.ProductivityCategory != 'MD&I', 'ProductivityCategory'] = 'Ops'
    prod=pd.DataFrame(pd.pivot_table(df, index='ProductivityCategory',columns=['Source'] ,values='Amount',aggfunc=sum).round().to_records())
    prod.loc[:,'YoY Var'] = prod['Forecast']- prod['Actuals']
    prod=prod.append({'ProductivityCategory':NetCost, 'Actuals':prod.Actuals.sum(),'Forecast': prod.Forecast.sum(),'YoY Var': prod['YoY Var'].sum()}, ignore_index=True)
    fig2 = go.Figure()
    # prod2 = prod

    #colors = ["rgb(150,150,150)","rgb(70,0,115)","rgb(0,142,255)"]
    if ((team!=[]) & (team4==[]) & (len(team)==1)  & (NetCost=='NetCost')):
        prod.loc[:,'Target']=0
        prod.loc[2,'Target']=targets.loc[team[0],'Target']
        prod=prod[['ProductivityCategory', 'Target', 'Forecast', 'Actuals', 'YoY Var']]
        colores = ['rgb(255,0,0)',"rgb(70,0,115)","rgb(150,150,150)","rgb(0,142,255)"]
        #print(prod)
        i = 0
        for each in prod.columns[1:]:
            # valor = "{:,}".format(str(prod[each]))
            fig2.add_trace(go.Bar(
            y=prod.ProductivityCategory,
            x=prod[each],
            name=each,
            orientation='h',
            marker_color=colores[i],
            # showgrid=True,
            text=[human_format(x) for x in prod[each]],
            textposition="auto",
                ))
            i+=1
    else:
        i = 0
        for each in prod.columns[1:]:
            # valor = "{:,}".format(str(prod[each]))
            fig2.add_trace(go.Bar(
            y=prod.ProductivityCategory,
            x=prod[each],
            name=each,
            orientation='h',
            marker_color=colors[i],
            # showgrid=True,
            text=[human_format(x) for x in prod[each]],
            textposition="auto",
                ))
            i+=1
        
        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig2.update_layout(title='Actual FY19 vs Forecast FY20 Plan YTD',plot_bgcolor='#FFFFFF',paper_bgcolor='#FFFFFF',font ={'color': '#7500c0'},barmode='group', xaxis_tickangle=-45,xaxis=dict(
        showgrid=True))

    ##Barras
    ActvsFct = pd.DataFrame(pd.pivot_table(df, index=['Source','FiscalMonthNumber','FiscalMonthName'], values='Amount',aggfunc=sum).round().to_records()).sort_values('FiscalMonthNumber')
    fig3 = go.Figure()
    i=0
    for each in ['Actuals','Forecast']:
        fig3.add_trace(go.Bar(name=each, x=ActvsFct[ActvsFct['Source']==each].FiscalMonthName, y=ActvsFct[ActvsFct['Source']==each].Amount,marker_color=colors[i]))
        i+=1

    fig3.update_layout(title='Actual FY19 vs Forecast FY20 Plan Trend',plot_bgcolor='#FFFFFF',paper_bgcolor='#FFFFFF',font ={'color': '#7500c0'})

    ##Table
    Data = pd.DataFrame(pd.pivot_table(df, columns='Source',index=['Number','ReportLine3'],values='Amount', aggfunc=sum).round().fillna(0).to_records())
    Data.loc[:,'YoY Var'] = (((Data['Forecast']-Data['Actuals'])/Data['Actuals']).mul(100).round(0)).astype(str) + '%'
    Data['Actuals'] = Data.apply(lambda x: "{:,}".format(x['Actuals']), axis=1)
    Data.Actuals.astype(str)
    Data.Actuals = Data.Actuals.str.split('.', expand=True)
    Data['Forecast'] = Data.apply(lambda x: "{:,}".format(x['Forecast']), axis=1)
    Data.Forecast.astype(str)
    Data.Forecast = Data.Forecast.str.split('.', expand=True)
    Data = Data.sort_values('Number')[['ReportLine3','Actuals','Forecast','YoY Var']]
    data=[go.Table(
                    columnwidth = [40,20,20,15],
                    header=dict(values=list(Data.columns),
                    line_color='darkslategray',
                    fill_color='royalblue',
                    align=['left','center'],
                    font=dict(color='white', size=12),
                    height=40
                ),
                    cells=dict(values=[Data.ReportLine3, Data.Actuals, Data.Forecast, Data['YoY Var']],
                    line_color='darkslategray',
                    fill=dict(color=['pink', 'white']),
                    align=['left', 'center'],
                    font_size=12,
                    height=30))]
    print(team, team4, start, time.time() - start)
    return  fig1,fig2, fig3, {'data':data}

@app.callback(
    [Output('graph-with-HC', 'figure'),
    Output('cost-class', 'figure'),
    Output('trend-with-HC', 'figure')
    ],
    [Input('team-column', 'value'), 
    Input('team-column-level4', 'value')])

def update_figure(team, team4):
    start = time.time()
    if team==[]:
        df=HC
    if ((len(team)>0) & (len(team4)>0)):
        print(team, team4)
        df = HC[np.in1d(HC.Level3Name,team) & np.in1d(HC.Level4Name,team4)]
        if df.shape[0]==0:
            return {'data': []},{'data': []},{'data': []}
    elif (len(team)>0):
        print(team, '2')
        df = HC[np.in1d(HC.Level3Name,team)]
        if df.shape[0]==0:
            return {'data': []},{'data': []},{'data': []}
    no_actuals = df[np.in1d(df.Source,'Actuals')].shape[0]
    no_forecast = df[np.in1d(df.Source,'Forecast')].shape[0]
    if (no_actuals==0) or (no_forecast==0):
        source = 'Actuals' if no_actuals==0 else 'Forecast'
        df2 = df.copy()
        df2.loc[:,'Amount']=0
        df2.loc[:,'Source']=source
        df= pd.concat([df, df2], ignore_index=True)
        # df.to_csv('test.csv')
        # return {'data': []},{'data': []},{'data': []}
    # meses_reales=df.FiscalMonthNumber.unique()
    # if len(df.FiscalMonthNumber.unique())<12:
    #     mes1=[]
    #     for each in months.FiscalMonthNumber:
    #         if each not in meses_reales:
    #             df=df.append({'ProductivityCategory':NetCost, 'Actuals':prod.Actuals.sum(),'Forecast': prod.Forecast.sum(),'YoY Var': prod['YoY Var'].sum()}, ignore_index=True)
    if team!=[]:
        emptyMonths['Level3Name'] = team[0]
        if team4!=[]:
            emptyMonths.Level4Name = team4[0]
        df= pd.concat([df, emptyMonths], ignore_index=True)
        
    yoy = df[df['FiscalMonthNumber']==12]
    ## levels
    HC1 = pd.pivot_table(yoy, index=['EmployeeLevelName'],columns=['Source'], values='Amount',aggfunc=sum).fillna(0).round()
    HC1=pd.DataFrame(HC1.reindex(np.flip(employeelevels.Name.unique())).fillna(0).to_records())
    lista= (HC1.Actuals.tolist() + HC1.Forecast.tolist())
    valor = max(lista)*1.1
    
    equis=[]
    
    valor1= valor
    for each in range(HC1.shape[1]):
        valor1 += -1 * (valor/3)
        equis.append(float(valor1))
        equis.append(float(-1 *valor1))
        
    equis.sort()
    equis1=[]
    for i, each in enumerate(equis):
        equis1.append(abs(int(each)))
        
    actuals_bins= HC1.Actuals*-1
    forecast_bins =HC1.Forecast
    
    
    y= HC1.EmployeeLevelName

    
    layout = go.Layout(plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
    #                 font= {
    #                     'color': '#7500c0'},
        
                    yaxis=go.layout.YAxis(title='Levels'),
                    xaxis=go.layout.XAxis(
                    range=[-1*valor, valor],
                    #tickvals=[-1*valor, valor],
                    ticktext=[int(valor), int(valor)],
                        title='HC'),
                    barmode='overlay',
                    bargap=0.1)
    
    data = [go.Bar(y=y,
                x=actuals_bins,
                orientation='h',
                name='Actuals',
                text=-1 * actuals_bins.astype('int'),
                # hoverinfo='text',
                textposition="auto",
                marker=dict(color="rgb(150,150,150)")
                ),
            go.Bar(y=y,
                x=forecast_bins,
                orientation='h',
                name='Forecast',
                text= forecast_bins.astype('int'),
                #hoverinfo='text',
    #                hoverinfo='x',
                marker=dict(color="rgb(70,0,115)"),
                #text=[("{:,}".format(x)).split('.')[0] for x in prod[each]],
                textposition="auto",)]
    fig1=go.Figure(data, layout)
    #donas costo
    Cost = pd.pivot_table(yoy,index=['Classification'], columns='Source',values='Amount', aggfunc=sum)
    labels = Cost.index
    colors = ["rgb(150,150,150)","rgb(70,0,115)"]
    #,"rgb(0,142,255)"]
    # Create subplots: use 'domain' type for Pie subplot
    fig2 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig2.add_trace(go.Pie(labels=labels, values=Cost['Actuals'].tolist(), name="FY19", marker_colors=colors),
                1, 1)
    fig2.add_trace(go.Pie(labels=labels, values=Cost['Forecast'].tolist(), name="FY20 Plan", marker_colors=colors),
                1, 2)

    # Use `hole` to create a donut-like pie chart
    fig2.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig2.update_layout(
        title_text="Cost Classification",plot_bgcolor='#FFFFFF',paper_bgcolor='#FFFFFF',font ={'color': '#7500c0'},
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='FY19', x=0.16, y=0.5, font_size=18, showarrow=False),
                    dict(text='FY20Plan', x=0.87, y=0.5, font_size=15, showarrow=False)])
    
    ## trend hc
    df = pd.DataFrame(pd.pivot_table(df, index=['FiscalMonthNumber','Source'], values='Amount',aggfunc=sum).round().to_records())

    title = 'HC Actual FY19 vs Forecast FY20 Trend'
    labels = ['Actuals', 'Forecast']
    colors = ['rgb(0,206,209)', 'rgb(138,43,226)']

    mode_size = [8, 12]
    line_size = [2, 3]

    x_data = np.vstack((months.FiscalMonthName.tolist(),)*2)

    y_data = np.array([df[df['Source']=='Actuals']['Amount'],df[df['Source']=='Forecast']['Amount']])

    fig = go.Figure()

    for i in range(0, 2):
        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
        ))

        # endpoints
        fig.add_trace(go.Scatter(
            x=[x_data[i][0], x_data[i][-1]],
            y=[y_data[i][0], y_data[i][-1]],
            mode='markers',
            marker=dict(color=colors[i], size=mode_size[i])
        ))

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white',
        font ={'color': '#7500c0'}
    )

    annotations = []

    # Adding labels
    for y_trace, label, color in zip(y_data, labels, colors):
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                    xanchor='right', yanchor='middle',
                                    text=label + ' {}'.format(y_trace[0]),
                                    font=dict(family='Arial',
                                                size=16),
                                    showarrow=False))
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],
                                    xanchor='left', yanchor='middle',
                                    text='{}'.format(y_trace[11]),
                                    font=dict(family='Arial',
                                                size=16),
                                    showarrow=False))
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text=title,
                                font=dict(family='Arial',
                                            size=24,
                                            color='rgb(138,43,226)'),
                                showarrow=False))

    fig.update_layout(annotations=annotations)
    print('hc',team, team4, start, time.time(), time.time() - start)
    return fig1, fig2, fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8084)