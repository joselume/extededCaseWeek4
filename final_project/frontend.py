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

# Libraries
def getDistributionFigure(field):
    # Define data sets
    benign_df = df_project[df_project['diagnosis']=='B'][field]
    malign_df  = df_project[df_project['diagnosis']=='M'][field]    
    hist_data = [benign_df, malign_df]
    group_labels = ['Benign', 'Malignant']

    # Create distplot with custom bin_size
    return ff.create_distplot(hist_data, group_labels, bin_size=.2)
    
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
                            dbc.Label("Concave points mean", html_for="concave-points-mean"),
                            dbc.Input(
                                type="number",
                                id="concave-points-mean",
                                placeholder="Enter concave points mean",
                            ),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Texture mean", html_for="texture-mean"),
                            dbc.Input(
                                type="number",
                                id="texture-mean",
                                placeholder="Enter texture mean",
                            ),
                        ]
                    ),
                    width=6,
                ),
                
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Area mean", html_for="area-mean"),
                            dbc.Input(
                                type="number",
                                id="area-mean",
                                placeholder="Enter area mean",
                            ),
                        ]
                    ),
                    width=6,
                ),   
                
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Compactness mean", html_for="compactness-mean"),
                            dbc.Input(
                                type="number",
                                id="compactness-mean",
                                placeholder="Enter compactness mean",
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
                    children=
                    [                    
                        dbc.Alert("The pacient is cancer free! Contratulations!", color="success"),                 
                    ],
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
                                    id="distribution1",
                                    figure=getDistributionFigure('radius_mean')                        
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
                                    id="distribution2",
                                    figure=getDistributionFigure('texture_mean')                                            
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
                                    id="distribution3",
                                    figure=getDistributionFigure('perimeter_mean')                        
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
                                    id="distribution4",
                                    figure=getDistributionFigure('area_mean')                                            
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

@app.callback(Output('output_div', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('concave-points-mean', 'value'),
              State('texture-mean', 'value'),
              State('area-mean', 'value'),
              State('compactness-mean', 'value')
              ],
                  ) 
def update_output(clicks, input_1, input_2, input_3, input_4):
    if clicks is not None:
        data = {'concave points_mean': input_1,
                'texture_mean': input_2,
                'area_mean': input_3,
                'compactness_mean':input_4}
        
        # convert data into dataframe
        data.update((x, [y]) for x, y in data.items())
        data_df = pd.DataFrame.from_dict(data)
        
    
        result = model.predict(data_df)
        
        if result == 0:
            return [                    
                        dbc.Alert("Negative diagnosis", color="success"),                 
                    ]
        elif result == 1:
            return [                    
                        dbc.Alert("Positive diagnosis", color="danger"),                 
                    ]
                    
        print('result: '+str(result))

if __name__ == "__main__":    
    app.run_server(host= '0.0.0.0', debug=True)
    
