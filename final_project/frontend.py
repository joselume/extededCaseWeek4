# X = df[[ 'perimeter_worst', 'radius_worst','concave points_worst', 'texture_worst']]
import pandas as pd
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_table
from sqlalchemy import create_engine

# Final project additional imports
import plotly.figure_factory as ff
import numpy as np
import dash_bootstrap_components as dbc # https://dash-bootstrap-components.opensource.faculty.ai/l/components/alert
import pickle

# Deployment. This deployment does not work in the part of Heroku, however I used to wrapped the model in the model.pkl:
# https://towardsdatascience.com/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50

###################################################################################################################################################

# load model
model = pickle.load(open('model_aws.pkl','rb'))

# Final project
engine = create_engine('postgresql://postgres:lNUEtV9XYCMUukxKvVKJ@final-project-db-machine.cjrnch5aefyf.us-east-1.rds.amazonaws.com/postgres')
df_project = pd.read_sql("SELECT * from fna_test", engine.connect())

print(df_project[['perimeter_worst', 'radius_worst','concave points_worst', 'texture_worst']].describe())

field1Desc = 'Perimeter worst (50.41 - 251.20) '
field2Desc = 'Radius worst (7.93 - 36.04) '
field3Desc = 'Concave points worst (0.000 - 0.291)'
field4Desc = 'Texture worst (12.02 - 49.54) '

popHeader1 = 'Perimeter worst'
popHeader2 = 'Radius worst'
popHeader3 = 'Concave points'
popHeader4 = 'Texture worst'

field1DescLower = 'perimeter worst'
field2DescLower = 'radius worst'
field3DescLower = 'concave points worst'
field4DescLower = 'texture worst'

popBody1 = 'Worst perimeter is the mean of the three largest distance around the nuclear border of all nuclei in the image.'
popBody2 = 'Worst radius is the mean of the three largest radial line segments from the center of the nuclear mass of all nuclei in the image.'
popBody3 = 'Worst concave points is the mean of the three largest number of points in the nuclear border that lie on an indentation of all nuclei in the image.'
popBody4 = 'Worst texture is the mean of the three largest variations of the gray scale intensities of all nuclei in the image.'

field1Name = 'perimeter_worst'
field2Name = 'radius_worst'
field3Name = 'concave points_worst'
field4Name = 'texture_worst'

field1DistPlotId = 'perimeterWorstDist'
field2DistPlotId = 'radiusWorstDist'
field3DistPlotId = 'concavePointsWorstDist'
field4DistPlotId = 'textureWorstDist'



# Libraries
def getDistributionFigure(field, label, xTupla, yTupla):
    # Define data sets
  
    benign_df = df_project[df_project['diagnosis']=='B'][field]
    malign_df  = df_project[df_project['diagnosis']=='M'][field]    
    hist_data = [benign_df, malign_df]
    group_labels = ['Benign', 'Malignant']
    #maxi=df_project.value_counts().iloc[0]
    #print(maxi)

    # Create distplot with custom bin_size
    fig=ff.create_distplot(hist_data, group_labels, bin_size=.2)
    fig.update_layout(title_text=label)
    if (xTupla is not None and yTupla is not None):
        fig.add_trace(go.Scatter(x=xTupla, y=yTupla, mode='lines', line=go.scatter.Line(color='red'), showlegend=False))
    return fig

###################################################################################################################################################

# Df from a local file
# df = pd.read_csv('aggr.csv', parse_dates=['Entry time'])

#app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/uditagarwal/pen/oNvwKNP.css', 'https://codepen.io/uditagarwal/pen/YzKbqyV.css'])
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

# lst = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
app.layout = html.Div(children=[
    
    dbc.Container([

        dbc.Row(dbc.Col(
            
            dbc.Jumbotron(
                [
                    html.H1("Breast Cancer Diagnosis Prediction", className="display-3"),
                    html.P(
                        "Final project done for the DS4A Colombian Program",                        
                        className="lead",
                    ),
                    html.P(
                        "Scientists: Karina Ceron"
                        ", Jose Luis Mesa Espinosa"
                        ", Cristhian Camilo Zamora"
                        ", and Santiago Gutiérrez"
                    ),                                        
                    html.Hr(className="my-2"),
                    html.P(
                        'Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The 3-dimensional space is described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].'),
                    html.P(
                    'To know more about the origin of the used dataset please visit the official website using the following link:'),
                                                                    
                    html.P(dbc.Button("Reference", color="primary", href='https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29'), className="lead"),
                ]
            )            
        )),  
        
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label(field1Desc, html_for=field1Name),
                            dbc.Button("?", id="popover-target-1", color="secondary", style={"margin":'10px', "height":"30px"}),
                            dbc.Popover(
                                [
                                    dbc.PopoverHeader(popHeader1),
                                    dbc.PopoverBody(popBody1),
                                ],
                                id="popover1",
                                is_open=False,
                                target="popover-target-1",
                            ),
                            dbc.Input(
                                type="number",
                                id=field1Name,
                                placeholder="Enter " + field1DescLower,
                            ),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label(field2Desc, html_for=field2Name),
                            dbc.Button("?", id="popover-target-2", color="secondary", style={"margin":'10px', "height":"30px"}),
                            dbc.Popover(
                                [
                                    dbc.PopoverHeader(popHeader2),
                                    dbc.PopoverBody(popBody2),
                                ],
                                id="popover2",
                                is_open=False,
                                target="popover-target-2",
                            ),
                            dbc.Input(
                                type="number",
                                id=field2Name,
                                placeholder="Enter " + field2DescLower,
                            ),
                        ]
                    ),
                    width=6,
                ),
                
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label(field3Desc, html_for=field3Name),
                            dbc.Button("?", id="popover-target-3", color="secondary", style={"margin":'10px', "height":"30px"}),
                            dbc.Popover(
                                [
                                    dbc.PopoverHeader(popHeader3),
                                    dbc.PopoverBody(popBody3),
                                ],
                                id="popover3",
                                is_open=False,
                                target="popover-target-3",
                            ),
                            dbc.Input(
                                type="number",
                                id=field3Name,
                                placeholder="Enter " + field3DescLower,
                            ),  
                            
                        ]
                    ),
                    width=6,
                ),                                                
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label(field4Desc, html_for=field4Name),
                            dbc.Button("?", id="popover-target-4", color="secondary", style={"margin":'10px', "height":"30px"}),
                            dbc.Popover(
                                [
                                    dbc.PopoverHeader(popHeader4),
                                    dbc.PopoverBody(popBody4)
                                ],
                                id="popover4",
                                is_open=False,
                                target="popover-target-4",
                            ),
                            dbc.Input(
                                type="number",
                                id=field4Name,
                                placeholder="Enter " + field4DescLower,
                            ),
                        ]
                    ),
                    width=6,
                ), 
                dbc.Col(
                    dbc.Button("Submit", color="primary", id ="submit-button"), 
                ),
                                                                       
            ],
            form=True,
        ),
        html.Br(),
        
        dbc.Row( 
            dbc.Col(
                
                html.Div(                      
                    id='output_div',                    
                ),
                width=12
            )
        ),

        dbc.Row(
            [
            dbc.Col( 
                html.Div(
                    className="row app-body",
                    children=[
                        html.Div(
                            className="twelve columns card",
                            children=[
                                dcc.Graph(
                                    id=field1DistPlotId,
                                    figure=getDistributionFigure(field1Name, field1Desc, None, None)                        
                                )
                            ]
                        )
                    ]
                ),
                width=6
            ),
            dbc.Col( 
                html.Div(
                    className="row app-body",
                    children=[
                        html.Div(
                            className="twelve columns card",
                            children=[
                                dcc.Graph(
                                    id=field2DistPlotId,
                                    figure=getDistributionFigure(field2Name, field2Desc, None, None)                                            
                                )
                            ]
                        )
                    ]
                ),
                width=6
            ),
            ]        
        ),    
        
        dbc.Row(
            [
            dbc.Col( 
                html.Div(
                    className="row app-body",
                    children=[
                        html.Div(
                            className="twelve columns card",
                            children=[
                                dcc.Graph(
                                    id=field3DistPlotId,
                                    figure=getDistributionFigure(field3Name, field3Desc, None, None)                        
                                )
                            ]
                        )
                    ]
                ),
                width=6
            ),
            dbc.Col( 
                html.Div(
                    className="row app-body",
                    children=[
                        html.Div(
                            className="twelve columns card",
                            children=[
                                dcc.Graph(
                                    id=field4DistPlotId,
                                    figure=getDistributionFigure(field4Name, field4Desc, None, None)                                            
                                )
                            ]
                        )
                    ]
                ),
                width=6
            ),
            ]        
        ),         
    ])
])        

# ====================================================================
# --- Callbacks --- #
'''
@app.callback(
    Output("concave-points-mean", "value"),
    Input("concave-points-mean", "value"),
)
def save_value(input_value):
    my_variable = input_value
    print(my_variable)
    return ''
'''

@app.callback(
    Output("popover1", "is_open"),
    [Input("popover-target-1", "n_clicks")],
    [State("popover1", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover2", "is_open"),
    [Input("popover-target-2", "n_clicks")],
    [State("popover2", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover3", "is_open"),
    [Input("popover-target-3", "n_clicks")],
    [State("popover3", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover4", "is_open"),
    [Input("popover-target-4", "n_clicks")],
    [State("popover4", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback([Output('output_div', 'children'),
               Output(field1DistPlotId, 'figure'),               
               Output(field2DistPlotId, 'figure'),
               Output(field3DistPlotId, 'figure'),
               Output(field4DistPlotId, 'figure')
              ],
              [Input('submit-button', 'n_clicks')],
              [State(field1Name, 'value'),
              State(field2Name, 'value'),
              State(field3Name, 'value'),
              State(field4Name, 'value')
              ],
                  ) 
def update_output(clicks, input_1, input_2, input_3, input_4):
    
    # Initializing the null values    
    if input_1 == None:
            input_1 = 0
    if input_2 == None:
            input_2 = 0 
    if input_3 == None:
            input_3 = 0 
    if input_4 == None:
            input_4 = 0
            
    # Setting the location for the feature indicators
    figField1 = getDistributionFigure(field1Name, field1Desc, [input_1, input_1], [0, 0.12]) 
    figField2 = getDistributionFigure(field2Name, field2Desc, [input_2, input_2], [0, 0.3])
    figField3 = getDistributionFigure(field3Name, field3Desc, [input_3, input_3], [0, 10]) 
    figField4 = getDistributionFigure(field4Name, field4Desc, [input_4, input_4], [0, 0.17])            

    if clicks is not None:
        data = {field1Name: input_1,
                field2Name: input_2,
                field3Name: input_3,
                field4Name: input_4}
        
        # convert data into dataframe
        data.update((x, [y]) for x, y in data.items())
        data_df = pd.DataFrame.from_dict(data)
            
        probatility = model.predict_proba(data_df)
        
        
        result = model.predict(data_df)
         
        if result == 0:
            prob = str(round(float(probatility[0][0])*100.0, 2))+"%"            
            return [                    
                        dbc.Alert("There is a probability of "+ prob + " that the tumor is BENIGN", color="success"),                 
                    ], figField1, figField2, figField3, figField4
        else:
            prob = str(round(float(probatility[0][1])*100.0, 2))+"%"
            return [                    
                        dbc.Alert("There is a probability of "+ prob + " that the tumor is MALIGNANT", color="danger"),     
                    ], figField1, figField2, figField3, figField4
    else:
        return [ ], figField1, figField2, figField3, figField4  

if __name__ == "__main__":    
    app.run_server(host= '0.0.0.0', debug=True)
