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

## **Social Determinant and User-Behavior Analysis**

# Joining screen_dict and replies datasets
merged_responsedata_raw = pd.merge(screen_dict, replies, on='response_id', how='outer', validate="many_to_many")

# Use boolean indexing to filter rows where 'column_name' is not 'main screen' as this is just the initial screen for every platform
merged_responsedata = merged_responsedata_raw[merged_responsedata_raw['response_theme'] != 'Main Screen']

# Extracting the 'year' and filtering the data
merged_responsedata['udate'] = pd.to_datetime(merged_responsedata['udate'])
merged_responsedata['year'] = merged_responsedata['udate'].dt.year

# Group the data by 'year' and 'value_name' and count the sessions
grouped_responsedata = merged_responsedata.groupby(['year', 'response_theme'])['sess_id'].count().unstack()

# Sort the columns by the sum of each column (total count of sessions) in descending order
grouped_responsedata = grouped_responsedata[grouped_responsedata.sum().sort_values(ascending=False).index]

# Define the dashboard
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Social Determinant and User Behavior Analysis Dashboard"),

    dcc.Dropdown(
        id='analysis-selection',
        options=[
            {'label': 'Response Themes by Year', 'value': 'response_theme'},
            {'label': 'Gender and Age Distribution', 'value': 'gender_age'},
            {'label': 'Gender and Procedure Commodity Category', 'value': 'gender_procedure_commodity'},
            {'label': 'Top 10 Commodity Products by Gender', 'value': 'top_commodities'},
            {'label': 'Gender and Age Range by Response Theme', 'value': 'age_range_response'}
        ],
        value='response_theme',
        multi=False,
        style={'backgroundColor': 'white', 'color': 'black'}
    ),

    html.Div(id='filter-options', children=[
        dcc.Checklist(
            id='gender-selection',
            options=[
                {'label': 'Male', 'value': 'Male'},
                {'label': 'Female', 'value': 'Female'}
            ],
            value=['Male', 'Female'],
            labelStyle={'display': 'inline-block', 'backgroundColor': 'white', 'color': 'black'}
        ),

        dcc.Dropdown(
            id='age-selection',
            options=[
                {'label': '<21', 'value': '<21'},
                {'label': '21-30', 'value': '21-30'},
                {'label': '31-40', 'value': '31-40'},
                {'label': '41-50', 'value': '41-50'},
                {'label': '51-60', 'value': '51-60'},
                {'label': '61-70', 'value': '61-70'}
            ],
            value=['<21', '21-30', '31-40', '41-50', '51-60', '61-70'],
            multi=True,
            style={'backgroundColor': 'white', 'color': 'black'}
        )
    ], style={'display': 'none'}),

    dcc.Graph(id='analysis-graph')
])

@app.callback(
    Output('filter-options', 'style'),
    [Input('analysis-selection', 'value')]
)
def toggle_filter_options(selected_analysis):
    if selected_analysis == 'age_range_response':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('analysis-graph', 'figure'),
    [Input('analysis-selection', 'value'),
     Input('gender-selection', 'value'),
     Input('age-selection', 'value')]
)
def update_graph(selected_analysis, selected_genders, selected_ages):
    if selected_analysis == 'response_theme':
        # Code for Response Themes by Year
        merged_responsedata['udate'] = pd.to_datetime(merged_responsedata['udate'])
        merged_responsedata['year'] = merged_responsedata['udate'].dt.year
        grouped_responsedata = merged_responsedata.groupby(['year', 'response_theme'])['sess_id'].count().unstack()
        grouped_responsedata = grouped_responsedata[grouped_responsedata.sum().sort_values(ascending=False).index[:10]]  # Select top 10 response themes
        fig = px.bar(grouped_responsedata, x=grouped_responsedata.index, y=grouped_responsedata.columns,
                     title='Sessions by Response Theme per Year', labels={'value': 'Total Sessions', 'index': 'Year'},
                     barmode='stack')

    elif selected_analysis == 'gender_age':
        # Code for Gender and Age Distribution
        age_df = databank[databank['key_name'] == 'age']
        gender_df = databank[databank['key_name'] == 'gender']
        merged_df = age_df.merge(gender_df, on='sess_id')
        grouped = merged_df.groupby('value_name_x')['value_name_y'].value_counts().unstack().fillna(0)
        fig = px.bar(grouped, x=grouped.index, y=grouped.columns,
                     title='Total Session Count by Gender and Age', labels={'value': 'Total Session Count', 'index': 'Age'},
                     barmode='stack')

    elif selected_analysis == 'gender_procedure_commodity':
        # Code for Gender and Procedure Commodity Category
        procedurecommoditycat_df = databank[databank['key_name'] == 'procedurecommoditycat']
        gender_df = databank[databank['key_name'] == 'gender']
        merged_df = procedurecommoditycat_df.merge(gender_df, on='sess_id')
        grouped = merged_df.groupby('value_name_x')['value_name_y'].value_counts().unstack().fillna(0)
        fig = px.bar(grouped, x=grouped.index, y=grouped.columns,
                     title='Total Session Count by Gender and Procedure Commodity Category', labels={'value': 'Total Session Count', 'index': 'Procedure Commodity Category'},
                     barmode='stack')

    elif selected_analysis == 'top_commodities':
        # Code for Top 10 Commodity Products by Gender
        commodityproduct_df = databank[databank['key_name'] == 'commodityproduct']
        gender_df = databank[databank['key_name'] == 'gender']
        merged_df = commodityproduct_df.merge(gender_df, on='sess_id')
        grouped = merged_df.groupby('value_name_x')['value_name_y'].value_counts().unstack().fillna(0)
        top_male = grouped['Male'].nlargest(10)
        top_female = grouped['Female'].nlargest(10)
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Male', 'Female'), specs=[[{'type': 'domain'}, {'type': 'domain'}]]) # type: ignore

        # Add male pie chart
        fig.add_trace(go.Pie(labels=top_male.index, values=top_male.values, name='Male'), 1, 1)

        # Add female pie chart
        fig.add_trace(go.Pie(labels=top_female.index, values=top_female.values, name='Female'), 1, 2)

        # Update layout
        fig.update_layout(title_text='Top 10 Commodity Products by Gender')

    elif selected_analysis == 'age_range_response':
        # Code for Gender and Age Range Distribution by Response Theme
        df = databank[databank['key_name'].isin(['gender', 'age'])]
        df = df.pivot_table(index='sess_id', columns='key_name', values='value_name', aggfunc='first')
        df = pd.merge(merged_responsedata, df, on='sess_id', how='outer', validate="many_to_many")
        df = df.dropna(subset=['age', 'gender', 'response_theme'], how='any')
        df = df[['age', 'gender', 'response_theme']]
        df = df.drop_duplicates(subset=['age', 'gender', 'response_theme'])

        # Filter data based on selected genders and ages
        df = df[df['gender'].isin(selected_genders) & df['age'].isin(selected_ages)]

        # Create the parallel categories diagram
        fig = go.Figure(go.Parcats(
            dimensions=[
                {'label': 'Response Theme', 'values': df['response_theme']},
                {'label': 'Gender', 'values': df['gender']},
                {'label': 'Age', 'values': df['age']}
            ],
            line={'color': 'blue'},
            hoveron='color',  # Only show tooltips for the boxes
            labelfont={'size': 12, 'family': 'Arial'},
            tickfont={'size': 10, 'family': 'Arial'},
            hoverinfo='count+probability',  # Show count and probability in the tooltip
            hovertemplate='%{label}<br>Count: %{count}<br>Probability: %{probability:.2f}<extra></extra>'
        ))

        fig.update_layout(title_text='Gender and Age Range Distribution by Response Theme')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)