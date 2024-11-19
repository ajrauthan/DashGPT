import openai
import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.express as px
import pandas as pd
import datetime as dt
import json
import subprocess
import os 
from google.cloud import bigquery
import google.api_core.exceptions
import re
import dash_bootstrap_components as dbc
import logging

# Set up logging configuration
logging.basicConfig(
    filename='app_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-cloud.json"

chat_history = []

# Define the path to the OpenAI credentials JSON file
openai_creds_path = "open_api_creds.json"

# Load the OpenAI API key from the JSON file
with open(openai_creds_path, 'r') as f:
    openai_credentials = json.load(f)

# Set the OpenAI API key
openai.api_key = openai_credentials["api_key"]

client_llm = openai.OpenAI(api_key=openai.api_key)

def GPTCompletion(messages, max_tokens=1000, temp=0.2):
    responses = client_llm.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        temperature=temp,
        messages=messages
    )
    response = responses.choices[0].message.content
    return response

# Initialize your BigQuery client
my_project_id = 'qst843-ajr'  
client_bq = bigquery.Client()
project_id = 'bigquery-public-data'
dataset_id = 'chicago_crime'


def get_table_schemas_with_descriptions(project_id, dataset_id):
    tables_query = f"""
    SELECT table_name
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
    WHERE table_type = 'BASE TABLE';
    """
    tables = client_bq.query(tables_query).result()
    table_schemas = {}
    for table in tables:
        table_name = table.table_name
        table_ref = client_bq.dataset(dataset_id, project=project_id).table(table_name)
        table_info = client_bq.get_table(table_ref)
        schema_info = table_info.schema
        table_schemas[table_name] = [
            (field.name, field.field_type, 'YES' if field.is_nullable else 'NO', field.description or '')
            for field in schema_info
        ]
    return table_schemas

schemas = get_table_schemas_with_descriptions(project_id, dataset_id)


def format_schemas(schemas):
    formatted_schemas = []
    for table, schema in schemas.items():
        table_info = {"table_name": table, "columns": []}
        for column in schema:
            column_info = {
                "column_name": column[0],
                "data_type": column[1],
                "is_nullable": column[2],
                "description": column[3]
            }
            table_info["columns"].append(column_info)
        formatted_schemas.append(table_info)
    return formatted_schemas

schemas = format_schemas(schemas)


def data_rows(project_id, dataset_id, limit=5):
    tables_query = f"""
    SELECT table_name
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
    WHERE table_type = 'BASE TABLE';
    """
    tables = client_bq.query(tables_query).result()
    table_data = {}
    for table in tables:
        table_name = table['table_name']
        sql_query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_name}`
        LIMIT {limit};
        """
        rows = client_bq.query(sql_query).to_dataframe()
        table_data[table_name] = rows
    return table_data

data = data_rows(project_id, dataset_id)


def query_gen(schema, data, user_query, chat_hist, project_id, dataset_id, retries=3):
    logging.info(f"User Query: {user_query}")
    system_prompt = f"""
    You are an expert in writing SQL queries based on data from BigQuery. 
    Given is the schema of the dataset here:\n{schema}.\n and the first few rows of the dataset here:\n{data}.
    You will be provided the schema of the dataset and the first few rows of the dataset. 
    Please do not create queries that reference something out of the schema.
    You are also an expert in converting a user's questions from English to a SQL query.
    Further, today's date is: {dt.date.today().strftime('%B %d, %Y')}. Please strictly use this information in case any time-relative questions are asked.
    
    If the dataset contains TIMESTAMP or DATETIME columns, prefer using DATE functions like DATE_SUB, DATE_ADD, 
    and DATE comparisons instead of TIMESTAMP functions, unless explicitly needed. 
    Try to avoid TIMESTAMP_SUB or TIMESTAMP_ADD and opt for DATE-based functions wherever possible to avoid compatibility issues. 
    Additionally, if time calculations are involved, consider converting TIMESTAMP or DATETIME to DATE when applicable."""
    
    prompt = f"""
    Please write a SQL query given the schema of the dataset here:\n{schema}.\n and the first few rows of the dataset here:\n{data}
    And the user query here:\n{user_query}\n
    The project ID is as follows:\n{project_id}\n and the dataset ID is as follows:\n{dataset_id}.
    Your response should only include the SQL query and nothing else.
    Please do not return any extra characters that could result in an error with the engine."""

    chat_hist.append({"role": "user", "content": user_query})
    messages = chat_hist + [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        GPT_SQL = GPTCompletion(messages, max_tokens=1000)
        if "```sql" in GPT_SQL:
            start_index = GPT_SQL.find("```sql") + len("```sql")
            end_index = GPT_SQL.find("```", start_index)
            GPT_SQL = GPT_SQL[start_index:end_index].strip()
        GPT_SQL = GPT_SQL.strip("`")
        logging.info(f"Generated SQL Query: {GPT_SQL}")
        df = client_bq.query(GPT_SQL).to_dataframe()
        logging.info(f"Query Result DataFrame:\n{df}")
        chat_hist.append({"role": "assistant", "content": f"Generated SQL query:\n{GPT_SQL}"})
        return df
    except google.api_core.exceptions.BadRequest as e:
        error_message = str(e)
        logging.error(f"Error occurred: {error_message}")
        return pd.DataFrame()


def gen_chat(result, user_query):
    logging.info(f"Data Summary Requested for Query: {user_query}")
    system_prompt = f"""
    You are an expert in interpreting results of SQL queries presented in a tabular data as a dataframe.
    Your are also an expert in relating the tabular data to the user query at hand and forming an appropriate answer comprising one or two lines."""
    
    prompt = f"""Given the following question of the user:\n{user_query}\n and the following results produced:\n{result} please compose a cohesive answer."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    summary = GPTCompletion(messages)
    logging.info(f"Generated Summary: {summary}")
    return summary


def gen_graph(result, user_query):
    logging.info(f"Data Visualization Requested for Query: {user_query}")
    system_prompt = f"""
    You are an expert in writing Python codes and interpreting data presented in a tabular data as a dataframe.
    Your are also an expert in relating the tabular data to the user query at hand and forming an appropriate visualization using plotly.express as px ."""
    
    prompt = f"""Given the following question of the user:\n{user_query}\n and the following results produced:\n{result} please create an appropriate visualization.
    Only give the Python code for creating the plot and nothing else.
    Do not create a synthetic database.
    This code would be stores in a variable later, so please make sure no extra information is a part of the output."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    code = GPTCompletion(messages)
    logging.info(f"Generated Plotly Graph Code: {code}")
    return code


def clean_code(code):
    logging.info(f"Data cleaning Requested for generated code:")
    prompt = f"""
    Extract only the line where px.FUNCTION() is used to generate a Plotly graph, and remove everything else. 
    Don't even include the variable assignment for px.FUNCTION(). 
    FUNCTION here could be any of the pc visualization functions - for example, bar(), line(), and so on.
    Provide only the function call with no surrounding code from the code below:
    {code}
    """
    messages = [{"role": "user", "content": prompt}]
    code = GPTCompletion(messages)
    if "```python" in code:
        start_index = code.find("```python") + len("```python")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    logging.info(f"Generated clean Graph Code: {code}")

    # Append cleaned graph code to chat history
    chat_history.append({"role": "assistant", "content": f"Generated Graph Code:\n{code}"})
    return code

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Adding a new button and collapsible section for chat history
app.layout = html.Div([
    html.H1("Data Query Chatbot and Visualization"),
    
    html.Div([
        html.Label("Ask your question:"),
        dcc.Textarea(id='question-area', value='', style={'width': '100%', 'height': 100}),
        html.Button('Submit', id='submit-btn', n_clicks=0),
        html.Button('Regenerate Summary', id='refresh-summary-btn', n_clicks=0)
    ]),

    # Add loading spinner for summary area
    dcc.Loading(
        id="loading-summary",
        type="circle",
        children=html.Div(id='summary-area', style={'margin-top': '20px'})
    ),

    # Text area for customizing graph
    html.Div([
        html.Label("Customize Graph:"),
        dcc.Textarea(id='customize-graph-area', value='', style={'width': '100%', 'height': 100}),
        html.Button('Apply Customization', id='apply-customization-btn', n_clicks=0)
    ], style={'margin-top': '20px'}),

    # Add loading spinner for graph area
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=dcc.Graph(id='graph-area', style={'margin-top': '20px'})
    ),

    # Button to toggle the chat history view
    html.Button("View Chat History", id="toggle-chat-history", n_clicks=0),

    # Button to clear the chat history
    html.Button("Clear Chat History", id="clear-chat-history", n_clicks=0, style={'margin-left': '10px'}),

    # Collapsible component for chat history (using dash_bootstrap_components)
    dbc.Collapse(
        html.Div(id='chat-history-area', style={"padding": "10px", "border": "1px solid #ddd", "margin-top": "10px"}),
        id='collapse-chat-history',
        is_open=False,
    ),
    
    dcc.Store(id='query-result-store'),  # Store the query result
    dcc.Store(id='graph-code-store')  # Store the graph code
])

# Function to format the chat history
def format_chat_history(chat_history):
    if not chat_history:
        return "No chat history yet."

    # Create a list of formatted entries showing role and content
    formatted_history = []
    for entry in chat_history:
        role = entry["role"].capitalize()
        content = entry["content"]
        formatted_history.append(html.Div(f"{role}: {content}", style={"margin-bottom": "10px"}))
    
    return formatted_history

# Callback to handle both toggling and clearing the chat history
@app.callback(
    [Output('collapse-chat-history', 'is_open'),
     Output('chat-history-area', 'children')],
    [Input('toggle-chat-history', 'n_clicks'),
     Input('clear-chat-history', 'n_clicks')],
    [State('collapse-chat-history', 'is_open')]
)
def handle_chat_history(toggle_clicks, clear_clicks, is_open):
    global chat_history
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Handle clear chat history button click
    if triggered == 'clear-chat-history' and clear_clicks > 0:
        chat_history = []  # Clear the chat history
        return is_open, format_chat_history(chat_history)  # Keep visibility as is, just clear the content

    # Handle toggle chat history button click
    elif triggered == 'toggle-chat-history' and toggle_clicks > 0:
        return not is_open, format_chat_history(chat_history)

    return is_open, format_chat_history(chat_history)

# Combined callback to handle submission and customization of the graph
@app.callback(
    [Output('query-result-store', 'data'),
     Output('summary-area', 'children'),
     Output('graph-area', 'figure'),
     Output('graph-code-store', 'data')],
    [Input('submit-btn', 'n_clicks'),
     Input('refresh-summary-btn', 'n_clicks'),
     Input('apply-customization-btn', 'n_clicks')],
    [State('question-area', 'value'),
     State('query-result-store', 'data'),
     State('customize-graph-area', 'value'),
     State('graph-code-store', 'data')]
)
def update_output(submit_clicks, refresh_summary_clicks, apply_customization_clicks, user_query, data, customization_text, graph_code):
    # Determine which button was clicked
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Handle submit button click
    if triggered == 'submit-btn' and submit_clicks > 0 and user_query:
        # Run the query and get the data
        df = query_gen(schemas, data, user_query, chat_history, project_id, dataset_id, retries=3)
        data_store = df.to_dict('records')
        
        # Generate the summary and graph
        summary = gen_chat(df, user_query)
        if not df.empty:
            # Generate the initial graph code based on user query and chat history
            x = gen_graph(df, user_query)
            x = clean_code(x)
            x = x.replace('\\n', ' ').replace('\\r', '').strip()
            x = re.sub(' +', ' ', x)
            exec_code = f"fig = {x}"
            context = {'df': df}
            exec(exec_code, globals(), context)
            fig = context.get('fig')
            # Store the initial cleaned graph code
            graph_code = x
        else:
            fig = {}
        return data_store, summary, fig, graph_code

    # Handle regenerate summary click
    elif triggered == 'refresh-summary-btn' and refresh_summary_clicks > 0 and data:
        df = pd.DataFrame(data)
        summary = gen_chat(df, user_query)
        return data, summary, dash.no_update, dash.no_update  # Only update the summary, not the graph

    # Handle apply customization click
    elif triggered == 'apply-customization-btn' and apply_customization_clicks > 0 and graph_code:
        # Generate a new graph code based on previous chat history and customization input
        prompt = f"""
        You are an expert in customizing Plotly visualizations based on user queries and prior context.
        Below is the chat history for reference:\n{chat_history}
        Use this context along with the following user customization request:\n"{customization_text}"
        The previously generated graph code is as follows:\n{graph_code}
        Generate the updated Plotly code incorporating the user's customization request.
        Only provide the modified Plotly code with no additional text.
        """
        
        # Use GPT model to get the new graph code with customization
        customized_code = GPTCompletion([{"role": "user", "content": prompt}])
        # Clean and streamline the new code
        customized_code = clean_code(customized_code)
        customized_code = customized_code.replace('\\n', ' ').replace('\\r', '').strip()
        customized_code = re.sub(' +', ' ', customized_code)

        # Execute the new customized graph code
        exec_code = f"fig = {customized_code}"
        context = {'df': pd.DataFrame(data)}
        exec(exec_code, globals(), context)
        fig = context.get('fig')
        return data, dash.no_update, fig, graph_code  # Only update the graph

    return None, "", {}, None

if __name__ == '__main__':
    app.run_server(debug=True)