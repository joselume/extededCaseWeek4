import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_table
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:ETug2fve0atGVfQItzDP@nops-demo-instance.cjrnch5aefyf.us-east-1.rds.amazonaws.com/strategy')
df = pd.read_sql("SELECT * from trades", engine.connect(), parse_dates=('Entry time',))

print(df.head())

# Df from a local file
# df = pd.read_csv('aggr.csv', parse_dates=['Entry time'])

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/uditagarwal/pen/oNvwKNP.css', 'https://codepen.io/uditagarwal/pen/YzKbqyV.css'])

app.layout = html.Div(children=[
    html.Div(
            children=[
                html.H2(children="Bitcoin Leveraged Trading Backtest Analysis", className='h2-title'),
            ],
            className='study-browser-banner row'
    ),
    html.Div(
        className="row app-body",
        children=[
            html.Div(
                className="twelve columns card",
                children=[
                    html.Div(
                        className="padding row",
                        children=[
                            html.Div(
                                className="two columns card",
                                children=[
                                    html.H6("Select Exchange",),
                                    dcc.RadioItems(
                                        id="exchange-select",
                                        options=[
                                            {'label': label, 'value': label} for label in df['Exchange'].unique()
                                        ],
                                        value='Bitmex',
                                        labelStyle={'display': 'inline-block'}
                                    )
                                ]
                            ),                            
                            html.Div(
                                id="two columns card 2",
                                className="two columns card",
                                children=[
                                    html.H6("Select leverage",),
                                    dcc.RadioItems(
                                        id="leverage-select",
                                        options=[
                                            {'label': label, 'value': label} for label in df['Margin'].unique()
                                        ],
                                        value=1,
                                        labelStyle={'display': 'inline-block'}
                                    )                                   
                                ]
                            ),
                            html.Div(
                                className="three columns card",
                                children=[
                                    html.H6("Select a Date Range",),
                                    dcc.DatePickerRange(
                                        id="date-range-select",
                                        start_date=df['Entry time'].min(),
                                        end_date=df['Entry time'].max(),
                                        display_format='MMM YY',
                                    )                                                                        
                                ]
                            ),                            
                            html.Div(
                                id="strat-returns-div",
                                className="two columns indicator pretty_container",
                                children=[
                                    html.P(id="strat-returns", className="indicator_value"),
                                    html.P('Strategy Returns', className="twelve columns indicator_text"),
                                ]
                            ),
                            html.Div(
                                id="market-returns-div",
                                className="two columns indicator pretty_container",
                                children=[
                                    html.P(id="market-returns", className="indicator_value"),
                                    html.P('Market Returns', className="twelve columns indicator_text"),
                                ]
                            ),
                            html.Div(
                                id="strat-vs-market-div",
                                className="two columns indicator pretty_container",
                                children=[
                                    html.P(id="strat-vs-market", className="indicator_value"),
                                    html.P('Strategy vs. Market Returns', className="twelve columns indicator_text"),
                                ]
                            ),                                                        
                        ]
                )
        ]),           
            
        html.Div(
            className="twelve columns card",
            children=[
                dcc.Graph(
                    id="monthly-chart",
                    figure=
                    {
                         'data': []
                    }
                    
                )
            ]
        ),               
        html.Div(
                className="padding row",
                children=[
                    html.Div(
                        className="six columns card",
                        children=[
                            dash_table.DataTable(
                                id='table',
                                columns=[
                                    {'name': 'Number', 'id': 'Number'},
                                    {'name': 'Trade type', 'id': 'Trade type'},
                                    {'name': 'Exposure', 'id': 'Exposure'},
                                    {'name': 'Entry balance', 'id': 'Entry balance'},
                                    {'name': 'Exit balance', 'id': 'Exit balance'},
                                    {'name': 'Pnl (incl fees)', 'id': 'Pnl (incl fees)'},
                                ],
                                style_cell={'width': '50px'},
                                style_table={
                                    'maxHeight': '450px',
                                    'overflowY': 'scroll'
                                },
                            )
                        ]
                    ),
                    
                    dcc.Graph(
                        id="pnl-types",
                        className="six columns card",
                        figure={},
                        style={'height':'450px', 'width': '800px'}
                    )
                ]
        ),
            
            html.Div(
                className="padding row",
                children=[
                    dcc.Graph(
                        id="daily-btc",
                        className="six columns card",
                        figure={},
                        style={'height':'450px', 'width': '1100px'}
                    ),
                    dcc.Graph(
                        id="balance",
                        className="six columns card",
                        figure={},
                        style={'height':'450px', 'width': '800px'}
                    )
                ]
            )
            
    ])        
])

######################
# Callback Functions #
######################

@app.callback(
    [
        dash.dependencies.Output('date-range-select', 'start_date'),
        dash.dependencies.Output('date-range-select', 'end_date')
    ],
    [
        dash.dependencies.Input('exchange-select', 'value'),
    ]
)
def update_date_range_select(exchange_value):
    return filter_date_range(exchange_value)


@app.callback(        
    [
        dash.dependencies.Output('monthly-chart', 'figure'),
        dash.dependencies.Output('market-returns', 'children'),
        dash.dependencies.Output('strat-returns', 'children'),
        dash.dependencies.Output('strat-vs-market', 'children'),
        dash.dependencies.Output('table', 'data'),
        dash.dependencies.Output('pnl-types', 'figure'),
        
    ],        
    [
        dash.dependencies.Input('exchange-select', 'value'),
        dash.dependencies.Input('leverage-select', 'value'),        
        dash.dependencies.Input('date-range-select', 'start_date'),
        dash.dependencies.Input('date-range-select', 'end_date'),
    ]
)
def calculate_monthly_returns (exchange, margin, start, end):        

    # Apply all the filters
    dff = filter_df (exchange, margin, start, end)
                    
    # Get the first entry and the last exit value per month
    df_entry, df_exit = getFirstEntryAndLastExitPerMonth(dff)   
    
    btc_returns = calc_btc_returns(dff)
    strat_returns = calc_strat_returns(dff)
    strat_vs_market = strat_returns - btc_returns        
    
    # Return the candlestick
    return {         
        'data': [{                
                'x': df_entry['yyyy-month'],
                'open': df_entry['Entry balance'],
                'high': df_exit['Exit balance'],
                'low': df_entry['Entry balance'],
                'close': df_exit['Exit balance'],
                'type': 'candlestick'
        }],
        'layout': {
            'title': 'Overview of Monthly Performance'
        }
    }, f'{btc_returns:0.2f}%', f'{strat_returns:0.2f}%', f'{strat_vs_market:0.2f}%', dff.to_dict('records'), print_bar_plot(dff)


def print_bar_plot (dff):
    
    long_df, short_df = get_bar_plot_dataframes(dff)
    
    return  go.Figure(
        data=[
            go.Bar(
                x=long_df['Entry time'],
                y=long_df['Pnl (incl fees)'],
                name='long',                
            ),
            go.Bar(
                x=short_df['Entry time'],
                y=short_df['Pnl (incl fees)'],
                name='short',                
            )
        ],
        layout=go.Layout(
            title='Pnl vs Trade Type',            
        )
    )


@app.callback(  
    [
        dash.dependencies.Output('daily-btc', 'figure'),
        dash.dependencies.Output('balance', 'figure'),
    ],
    [
        dash.dependencies.Input('date-range-select', 'start_date'),
        dash.dependencies.Input('date-range-select', 'end_date'),
    ]
)
def upd_btc_and_balance(start, end):
    
    dff = apply_filter(df.copy(), 'Entry time', start, 'greater')    
    dff = apply_filter(dff, 'Entry time', end, 'less') 
    
    return update_btc_price(dff), update_balance(dff) 

def update_btc_price(dff):    
    
    return go.Figure(
        data=[
            go.Scatter(x=dff['Entry time'], y=dff['BTC Price'])            
        ],
        layout=go.Layout(
            title='Daily BTC Price',            
        )
    )

def update_balance(dff):
    return go.Figure(
        data=[
            go.Scatter(x=dff['Entry time'], y=dff['Exit balance'] - dff['Entry balance'])
        ],
        layout=go.Layout(
            title='Portfolio Balance',
        )
    )

####################
# Helper Functions #
####################

def filter_df (exchange, margin, start, end):
    dff = apply_filter(df.copy(), 'Exchange', exchange, 'equal')    
    dff = apply_filter(dff, 'Margin', margin, 'equal')    
    dff = apply_filter(dff, 'Entry time', start, 'greater')    
    dff = apply_filter(dff, 'Entry time', end, 'less') 
    return dff

def getFirstEntryAndLastExitPerMonth(dff):
    dff['yyyy-month'] = dff['Entry time'].dt.strftime('%Y-%m')    
    df_grouped = dff.sort_values('Entry time', ascending=True) .groupby('yyyy-month', sort=False) 
    df_entry = df_grouped['Entry balance'].first().reset_index()
    df_exit = df_grouped['Exit balance'].last().reset_index()
    return df_entry, df_exit

def filter_date_range(exchange_value):
    dff = apply_filter(df.copy(), 'Exchange', exchange_value, 'equal')    
    return dff['Entry time'].min(), dff['Entry time'].max()

def apply_filter(df_to_filter, key, value, operator):
    if operator == 'equal':
        return df_to_filter[df_to_filter[key] == value]    
    if operator == 'less':
        return df_to_filter[df_to_filter[key] <= value]    
    if operator == 'greater':
        return df_to_filter[df_to_filter[key] >= value]    
                            
def calc_returns_over_month(dff):
    out = []

    for name, group in dff.groupby('YearMonth'):
        exit_balance = group.head(1)['Exit balance'].values[0]
        entry_balance = group.tail(1)['Entry balance'].values[0]
        monthly_return = (exit_balance*100 / entry_balance)-100
        out.append({
            'month': name,
            'entry': entry_balance,
            'exit': exit_balance,
            'monthly_return': monthly_return
        })
    return out

def calc_btc_returns(dff):
    btc_start_value = dff.tail(1)['BTC Price'].values[0]
    btc_end_value = dff.head(1)['BTC Price'].values[0]
    btc_returns = (btc_end_value * 100/ btc_start_value)-100
    return btc_returns

def calc_strat_returns(dff):
    start_value = dff.tail(1)['Exit balance'].values[0]
    end_value = dff.head(1)['Entry balance'].values[0]
    returns = (end_value * 100/ start_value)-100
    return returns             

def get_bar_plot_dataframes (dff):    
    dff_copy = dff.copy()    
    
    # Grouping by year and week
    dff_copy['Entry time'] = dff_copy['Entry time'].dt.strftime('%Y-%U')        
    dff_long = dff_copy[dff_copy['Trade type']=='Long'].groupby('Entry time').sum().reset_index()
    dff_short = dff_copy[dff_copy['Trade type']=='Short'].groupby('Entry time').sum().reset_index()
    
    # Without grouping
    # dff_long = dff_copy[dff_copy['Trade type']=='Long']
    # dff_short = dff_copy[dff_copy['Trade type']=='Short']    
        
    return dff_long, dff_short

if __name__ == "__main__":    
    app.run_server(host= '0.0.0.0', debug=True)
    
