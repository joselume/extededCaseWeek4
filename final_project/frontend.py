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
model = pickle.load(open('model.pkl','rb'))

# Final project
engine = create_engine('postgresql://postgres:lNUEtV9XYCMUukxKvVKJ@final-project-db-machine.cjrnch5aefyf.us-east-1.rds.amazonaws.com/postgres')
df_project = pd.read_sql("SELECT * from fna_test", engine.connect())

field1Desc = 'Perimeter worst'
field2Desc = 'Radius worst'
field3Desc = 'Concave points worst'
field4Desc = 'Texture worst'

field1DescLower = 'perimeter worst'
field2DescLower = 'radius worst'
field3DescLower = 'concave points worst'
field4DescLower = 'texture worst'

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
        fig.add_trace(go.Scatter(x=xTupla, y=yTupla, mode='lines', line=go.scatter.Line(color='grey'), showlegend=False))
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
                    html.H1("Breast cancer Diagnosis Prediction", className="display-3"),
                    html.P(
                        "Final project done for the DS4A Colombian program",                        
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
                        'Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].'),
                    html.P(
                    'To know more about the origin of this data set please visit the official website using the following link:'),
                                                                    
                    html.P(dbc.Button("Official website", color="primary", href='https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29'), className="lead"),
                ]
            )            
        )),  
        
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label(field1Desc, html_for=field1Name),
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
                            dbc.Input(
                                type="number",
                                id=field4Name,
                                placeholder="Enter " + field4DescLower,
                            ),
                        ]
                    ),
                    width=6,
                ), 
                dbc.Button("Submit", color="primary", id ="submit-button"),                                                        
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
    figField3 = getDistributionFigure(field3Name, field3Desc, [input_3, input_3], [0, 0.05]) 
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