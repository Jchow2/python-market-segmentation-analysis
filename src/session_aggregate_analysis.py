import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from datetime import datetime

import os
from pathlib import Path

# Set the working directory to the data/raw folder relative to the script location
script_dir = Path(__file__).resolve().parent
data_dir = (script_dir / '../../project-root/data/raw').resolve()

# Convert the path to a string and replace backslashes with forward slashes
data_dir_str = str(data_dir).replace('\\', '/')
os.chdir(data_dir_str)

# Access individual DataFrames
# Read all five platform datasets in the sauti databank
databank = pd.read_csv('platform_databank.csv')
sessions = pd.read_csv('platform_sessions.csv')
replies = pd.read_csv('platform_replies.csv')
screen_dict = pd.read_csv('platform_screen_dict.csv')
requests = pd.read_csv('platform_requests.csv')

import dash # type: ignore
from dash import dcc, html # type: ignore
from dash.dependencies import Input, Output # type: ignore
import plotly.express as px # type: ignore
from matplotlib.colors import rgb2hex # type: ignore

## **Session-Based Analysis**

# Parse the 'created_date' column in the databank dataset
databank['created_date'] = pd.to_datetime(databank['created_date'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
# Drop rows with NaT values in 'created_date'
databank.dropna(subset=['created_date'], inplace=True)
# Extract the year from 'created_date'
databank['year'] = databank['created_date'].dt.year
# Filter rows with 'key_name' as 'border' in the databank dataset
filtered_databank = databank[databank['key_name'] == 'border']

# Get unique values in 'value_name' column
databank.dropna(subset=['created_date'], inplace=True)
databank['year'] = databank['created_date'].dt.year

# Define the dashboard
app = dash.Dash(__name__)

# Define the dashboard
app.layout = html.Div([
    html.H1("Sauti Market Session-Based Analysis Dashboard"),

    dcc.Dropdown(
        id='variable-selection',
        options=[
            {'label': 'Procedure Commodity Category', 'value': 'procedurecommoditycat'},
            {'label': 'Commodity Meta Category', 'value': 'commoditymetacat'},
            {'label': 'Procedure Required Document', 'value': 'procedurerequireddocument'},
            {'label': 'Procedure Relevant Agency', 'value': 'procedurerelevantagency'}
        ],
        value='procedurecommoditycat',
        multi=False
    ),

    dcc.Graph(id='stacked-bar-chart')
])

@app.callback(
    Output('stacked-bar-chart', 'figure'),
    [Input('variable-selection', 'value')]
)
def update_graph(selected_variable):
    filtered_databank = databank[databank['key_name'] == selected_variable]

    # Define a threshold for significant categories
    threshold = 100  # Adjust this value as needed
    category_counts = filtered_databank['value_name'].value_counts()
    significant_categories = category_counts[category_counts > threshold].index

    # Filter out smaller categories
    filtered_databank = filtered_databank[filtered_databank['value_name'].isin(significant_categories)]

    pivot_databank = filtered_databank.pivot_table(index='year', columns='value_name', values='sess_id', aggfunc='count')

    # Convert RGB tuples to hex color strings using matplotlib.colors.rgb2hex
    colors = sns.color_palette("viridis", len(pivot_databank.columns))
    hex_colors = [rgb2hex(color) for color in colors]

    # Create the stacked bar chart
    fig = px.bar(pivot_databank.reset_index(),
                 x='year',
                 y=pivot_databank.columns,
                 title=f'Session Count by {selected_variable.capitalize()} per Year',
                 labels={'value': 'Session Count', 'year': 'Year', 'variable': 'Categories'},
                 barmode='stack',
                 color_discrete_sequence=hex_colors)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)