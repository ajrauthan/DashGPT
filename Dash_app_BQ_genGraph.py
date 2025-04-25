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
from config.config_loader import ConfigLoader
import plotly.graph_objects as go

# Load configuration
config = ConfigLoader()

# Set up logging configuration
logging_config = config.get_logging_config()
logging.basicConfig(
    filename=logging_config['filename'],
    level=logging_config['level'],
    format=logging_config['format'],
    datefmt=logging_config['date_format']
)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable 
credentials_config = config.get_credentials_config()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_config['google_cloud_json']

chat_history = []

# Load the OpenAI API key from the JSON file
with open(credentials_config['openai_creds_json'], 'r') as f:
    openai_credentials = json.load(f)

# Set the OpenAI API key
openai.api_key = openai_credentials["api_key"]

client_llm = openai.OpenAI(api_key=openai.api_key)

def GPTCompletion(messages, max_tokens=None, temp=None):
    openai_config = config.get_openai_config()
    responses = client_llm.chat.completions.create(
        model=openai_config['model'],
        max_tokens=max_tokens or openai_config['max_tokens'],
        temperature=temp or openai_config['temperature'],
        messages=messages
    )
    response = responses.choices[0].message.content
    return response

# Initialize your BigQuery client
client_bq = bigquery.Client()
bigquery_config = config.get_bigquery_config()
project_id = bigquery_config['project_id']
dataset_id = bigquery_config['dataset_id']


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


def data_rows(project_id, dataset_id, limit=3):
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
    You are an expert in writing SQL queries based on data from BigQuery dataset and a user's questions. 
    You will be provided the schema and the first few rows of the dataset. Please strictly respond to the user's query and follow the rules below:

    IMPORTANT FORMATTING RULES:
    1. Always use proper backtick (`) formatting for BigQuery table references
    2. Table references should always be in the format: `project_id.dataset_id.table_name`
    3. Every backtick must be closed
    4. Do not use single or double quotes for table names, only backticks

    QUERY CONTENT RULES:
    1. Only reference tables and columns that exist in the provided schema
    2. Do not create queries that reference anything outside of the schema
    3. Today's date is: {dt.date.today().strftime('%B %d, %Y')} - use this for time-relative queries

    DATETIME HANDLING RULES:
    1. For TIMESTAMP or DATETIME columns:
    - Prefer DATE functions (DATE_SUB, DATE_ADD, DATE) over TIMESTAMP functions
    - Avoid TIMESTAMP_SUB or TIMESTAMP_ADD
    - Convert TIMESTAMP/DATETIME to DATE when possible for calculations
    2. Example of correct datetime handling:
    - Use: DATE(timestamp_column)
    - Use: DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    - Avoid: TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)

    VALIDATION CHECKLIST:
    1. All table references use backticks
    2. Full table references include project_id and dataset_id
    3. Only reference existing schema elements
    4. Proper DATE functions used for time calculations

    Your response should contain ONLY the SQL query, with no additional text or formatting.
    """
    
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

    for attempt in range(retries):
        try:
            # Generate SQL query using configuration
            openai_config = config.get_openai_config()
            GPT_SQL = GPTCompletion(messages, max_tokens=openai_config['max_tokens'])
            if "```sql" in GPT_SQL:
                start_index = GPT_SQL.find("```sql") + len("```sql")
                end_index = GPT_SQL.find("```", start_index)
                GPT_SQL = GPT_SQL[start_index:end_index].strip()
            GPT_SQL = GPT_SQL.strip("`")
            
            # Try to run the query
            logging.info(f"Attempting to run query (attempt {attempt + 1}/{retries}): {GPT_SQL}")
            df = client_bq.query(GPT_SQL).to_dataframe()
            
            # Check if the result is empty
            if df.empty:
                logging.warning(f"Query returned empty results (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    error_feedback = f"""
                    The previous query returned no results. Please modify the query to:
                    1. Broaden any time-based filters
                    2. Remove overly restrictive conditions
                    3. Check for NULL values
                    4. Consider using LEFT JOIN instead of INNER JOIN
                    5. Use COALESCE for NULL handling
                    
                    Original query: {GPT_SQL}
                    """
                    messages.append({"role": "user", "content": error_feedback})
                    continue
                else:
                    logging.error("All query attempts returned empty results")
                    return pd.DataFrame()
            
            logging.info(f"Query executed successfully. Result DataFrame:\n{df}")
            # Format SQL query with proper syntax highlighting
            formatted_sql = f"```sql\n{GPT_SQL}\n```"
            chat_hist.append({"role": "assistant", "content": formatted_sql})
            return df
            
        except Exception as e:
            error_message = str(e)
            logging.warning(f"Query execution failed (attempt {attempt + 1}/{retries}): {error_message}")
            
            # If this was the last attempt, return empty DataFrame
            if attempt == retries - 1:
                logging.error("All query attempts failed")
                return pd.DataFrame()
            
            # Add error feedback to the messages for the next attempt
            error_feedback = f"""
            The previous query generated an error: {error_message}
            Please fix the query and try again. Pay special attention to:
            1. Table and column names
            2. SQL syntax
            3. Backtick formatting
            4. Date/time function usage
            """
            messages.append({"role": "user", "content": error_feedback})
    
    return pd.DataFrame()


def gen_chat(result, user_query):
    logging.info(f"Data Summary Requested for Query: {user_query}")
    system_prompt = f"""
    You are an expert in interpreting results of SQL queries presented in a tabular data as a dataframe.
    Your are also an expert in relating the tabular data to the user query at hand and forming an appropriate answer comprising one or two lines.
    When using numbers, please use the numbers in the results and do not make up new ones. 
    Also, please use approximate numbers when discussing high level data. Include the units of measurement when appropriate."""
    
    prompt = f"""Given the following question of the user:\n{user_query}\n and the following results produced:\n{result} please compose a cohesive answer."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    summary = GPTCompletion(messages)
    logging.info(f"Generated Summary: {summary}")
    return summary


def validate_visualization_code(code: str, df: pd.DataFrame) -> tuple[bool, str, str]:
    """
    Validates the visualization code and ensures it uses the correct data.
    
    Args:
        code (str): The visualization code to validate
        df (pd.DataFrame): The DataFrame to be used in visualization
        
    Returns:
        tuple[bool, str, str]: (is_valid, error_message, cleaned_code)
    """
    try:
        # Basic code safety checks
        if not code.strip():
            return False, "Empty visualization code", ""
            
        # Ensure the code uses the provided DataFrame
        if 'df' not in code:
            return False, "Code must use the provided DataFrame 'df'", ""
            
        # Clean and prepare the code
        code = code.replace('\\n', ' ').replace('\\r', '').strip()
        code = re.sub(' +', ' ', code)
        
        logging.info(f"Attempting to execute visualization code: {code}")
        
        # Try to execute the code in the full Python environment
        local_vars = {'df': df, 'px': px}
        exec(f"fig = {code}", globals(), local_vars)
        
        # Check if fig was created and is a valid Plotly figure
        fig = local_vars.get('fig')
        if not fig:
            return False, "No figure was created", ""
            
        # Validate the figure
        if not isinstance(fig, (go.Figure, dict)):
            return False, "Invalid figure object created", ""
            
        # Check if the figure uses the correct data
        if hasattr(fig, 'data') and fig.data:
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    # Check if x is a string (column name) or a Series/array
                    if isinstance(trace.x, str):
                        if trace.x not in df.columns:
                            return False, f"Column '{trace.x}' not found in DataFrame", ""
                if hasattr(trace, 'y') and trace.y is not None:
                    # Check if y is a string (column name) or a Series/array
                    if isinstance(trace.y, str):
                        if trace.y not in df.columns:
                            return False, f"Column '{trace.y}' not found in DataFrame", ""
        
        logging.info("Visualization code executed successfully")
        return True, "", code
        
    except Exception as e:
        error_message = str(e)
        logging.error(f"Visualization validation failed: {error_message}")
        logging.error(f"Failed code: {code}")
        return False, error_message, ""

def gen_graph(result, user_query):
    logging.info(f"Data Visualization Requested for Query: {user_query}")
    system_prompt = f"""
    You are an expert in writing Python codes and interpreting data presented in a tabular data as a dataframe.
    Your are also an expert in relating the tabular data to the user query at hand and forming an appropriate visualization using plotly.express as px.
    
    IMPORTANT RULES:
    1. Always use the provided DataFrame 'df' directly
    2. Do not create new DataFrames or modify the existing one
    3. Use only columns that exist in the DataFrame
    4. Keep the visualization code simple and focused
    5. Always return a valid Plotly figure
    6. Do not include any imports or variable assignments
    7. Only provide the plotly express function call
    """
    
    # Get sample data for the prompt
    sample_data = result.head().to_string()
    
    prompt = f"""Given the following question of the user:\n{user_query}\n and the following results produced:\n{result} please create an appropriate visualization.
    
    The DataFrame is named 'df' and contains the following columns: {list(result.columns)}.
    
    Here are the first few rows of the data:
    {sample_data}
    
    Only give the Python code for creating the plot and nothing else.
    Do not create a synthetic database.
    This code would be stored in a variable later, so please make sure no extra information is part of the output.
    
    Example of correct format:
    px.line(df, x='year', y='ride_count', title='Rides Over Time')"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            code = GPTCompletion(messages)
            logging.info(f"Raw generated code (attempt {attempt + 1}): {code}")
            
            if "```python" in code:
                start_index = code.find("```python") + len("```python")
                end_index = code.find("```", start_index)
                code = code[start_index:end_index].strip()
            
            logging.info(f"Cleaned code (attempt {attempt + 1}): {code}")
            
            # Validate the visualization code
            is_valid, error_message, cleaned_code = validate_visualization_code(code, result)
            
            if is_valid:
                logging.info(f"Generated valid visualization code: {cleaned_code}")
                return cleaned_code
            else:
                logging.warning(f"Visualization validation failed (attempt {attempt + 1}/{max_attempts}): {error_message}")
                
                # Add error feedback to the messages for the next attempt
                error_feedback = f"""
                The previous visualization code generated an error: {error_message}
                Please fix the code and try again. Pay special attention to:
                1. Using only the provided DataFrame 'df'
                2. Using only existing columns: {list(result.columns)}
                3. Creating a valid Plotly figure
                4. Following the example format exactly
                """
                messages.append({"role": "user", "content": error_feedback})
                
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in visualization generation (attempt {attempt + 1}/{max_attempts}): {error_message}")
            if attempt == max_attempts - 1:
                raise
    
    raise Exception("Failed to generate valid visualization code after multiple attempts")


def clean_code(code):
    logging.info(f"Data cleaning Requested for generated code:")
    prompt = f"""
    Extract only the plotly express function call from the code below.
    Do not include any imports, variable assignments, or additional code.
    Only return the px.FUNCTION() call that creates the visualization.
    Do not include fig.show() or any DataFrame modifications.
    
    Example of correct output:
    px.line(df, x='date', y='ride_count', title='Number of Rides per Month')
    
    Code to clean:
    {code}
    """
    messages = [{"role": "user", "content": prompt}]
    code = GPTCompletion(messages)
    if "```python" in code:
        start_index = code.find("```python") + len("```python")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    
    # Remove any fig.show() calls
    code = code.replace("fig.show()", "").strip()
    
    # Remove any DataFrame modifications
    if "df[" in code:
        # Extract only the plotly express call
        code = code.split("fig = ")[-1].strip()
    
    logging.info(f"Generated clean Graph Code: {code}")

    # Append cleaned graph code to chat history
    formatted_code = f"```python\n{code}\n```"
    chat_history.append({"role": "assistant", "content": formatted_code})
    return code

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Import the layout
from components.layout import create_layout

# Set the app layout
app.layout = create_layout()

# Function to format the chat history
def format_chat_history(chat_history):
    if not chat_history:
        return "No chat history yet."

    # Create a list of formatted entries showing role and content
    formatted_history = []
    for entry in chat_history:
        role = entry["role"].capitalize()
        content = entry["content"]
        
        # Format the content with markdown for proper code highlighting
        formatted_content = dcc.Markdown(
            content,
            style={
                'backgroundColor': 'white',
                'padding': '10px',
                'borderRadius': '5px',
                'border': '1px solid #dee2e6'
            }
        )
        
        formatted_history.append(html.Div([
            html.Strong(f"{role}: "),
            formatted_content
        ], style={"margin-bottom": "20px"}))
    
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

# Callback to handle immediate UI updates when submit is clicked
@app.callback(
    Output('customize-graph-area', 'value'),
    Input('submit-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_customization_textarea(submit_clicks):
    if submit_clicks and submit_clicks > 0:
        return ""
    return dash.no_update

# Callback to handle query submission and summary generation
@app.callback(
    [Output('query-result-store', 'data'),
     Output('summary-area', 'children'),
     Output('graph-code-store', 'data'),
     Output('graph-area', 'figure')],
    [Input('submit-btn', 'n_clicks')],
    [State('question-area', 'value')],
    prevent_initial_call=True
)
def update_query_and_summary(submit_clicks, user_query):
    if submit_clicks > 0 and user_query:
        global data  # to use the global data object
        df = query_gen(schemas, data, user_query, chat_history, project_id, dataset_id, retries=3)
        data_store = df.to_dict('records')
        summary = gen_chat(df, user_query)
        # Add the summary to chat history
        chat_history.append({"role": "assistant", "content": summary})
        if not df.empty:
            x = gen_graph(df, user_query)
            x = clean_code(x)
            x = x.replace('\\n', ' ').replace('\\r', '').strip()
            x = re.sub(' +', ' ', x)
            graph_code = x
            # Create the figure here
            exec_code = f"fig = {x}"
            context = {'df': df}
            exec(exec_code, globals(), context)
            fig = context.get('fig')
        else:
            graph_code = ""
            fig = {}
        return data_store, summary, graph_code, fig

    return None, "", "", {}

# Separate callback for regenerating summary
@app.callback(
    Output('summary-area', 'children', allow_duplicate=True),
    [Input('refresh-summary-btn', 'n_clicks')],
    [State('question-area', 'value'),
     State('query-result-store', 'data')],
    prevent_initial_call=True
)
def regenerate_summary(refresh_clicks, user_query, data):
    if refresh_clicks > 0 and data:
        df = pd.DataFrame(data)
        summary = gen_chat(df, user_query)
        # Add the new summary to chat history
        chat_history.append({"role": "assistant", "content": summary})
        return summary
    return dash.no_update

# Update the graph callback to only handle customization
@app.callback(
    Output('graph-area', 'figure', allow_duplicate=True),
    [Input('apply-customization-btn', 'n_clicks')],
    [State('query-result-store', 'data'),
     State('customize-graph-area', 'value'),
     State('graph-code-store', 'data')],
    prevent_initial_call=True
)
def update_graph(apply_customization_clicks, data, customization_text, graph_code):
    if apply_customization_clicks > 0 and graph_code:
        logging.info(f"Customization button clicked with text: {customization_text}")
        logging.info(f"Current graph code: {graph_code}")
        
        # Add the customization request to chat history
        global chat_history
        chat_history.append({"role": "user", "content": f"{customization_text}"})
        logging.info(f"Added customization request to chat history. New length: {len(chat_history)}")
        
        success, fig, error_message = validate_customization(customization_text, graph_code, data, chat_history)
        if success:
            logging.info("Customization successful")
            chat_history.append({"role": "assistant", "content": f"```python\n{graph_code}\n```"})
            return fig
        else:
            logging.error(f"Customization failed: {error_message}")
            return {}
    return {}

def validate_customization(customization_text, graph_code, data, chat_history, max_attempts=3):
    """
    Validates and retries customization attempts until a valid visualization is generated.
    
    Args:
        customization_text (str): User's customization request
        graph_code (str): Original graph code
        data (list): DataFrame data as list of records
        chat_history (list): List of previous chat messages for context
        max_attempts (int): Maximum number of retry attempts
        
    Returns:
        tuple: (success, figure, error_message)
    """
    logging.info(f"Validating customization request: {customization_text}")
    logging.info(f"Current chat history length: {len(chat_history)}")
    
    # Extract previous customization requests from chat history
    previous_customizations = []
    for msg in chat_history:
        if msg["role"] == "user" and "customize" in msg["content"].lower():
            previous_customizations.append(msg["content"])
            logging.info(f"Found previous customization: {msg['content']}")
    
    logging.info(f"Found {len(previous_customizations)} previous customizations")
    
    prompt = f"""
    You are an expert in customizing Plotly visualizations based on user queries and prior context.
    
    Previous customization requests (in chronological order):
    {chr(10).join(previous_customizations)}
    
    Current customization request: {customization_text}
    
    The previously generated graph code is as follows:
    {graph_code}
    
    The DataFrame is named 'df' and contains the following columns: {list(pd.DataFrame(data).columns)}.
    
    IMPORTANT RULES:
    1. Consider all previous customization requests when generating the new visualization
    2. Maintain any valid customizations from previous requests (e.g., if user asked for green color, keep it)
    3. Only modify aspects specifically requested in the current customization
    4. Ensure the final visualization combines all valid customization requests
    5. ONLY return the plotly express function call (e.g., px.bar(...)) without any imports or fig.show()
    6. Do not include any variable assignments or additional code
    
    Generate the updated Plotly code incorporating all customization requests.
    Only provide the modified Plotly code with no additional text.
    """
    
    logging.info(f"Generated prompt for customization")
    
    for attempt in range(max_attempts):
        try:
            # Generate customized code
            customized_code = GPTCompletion([{"role": "user", "content": prompt}])
            logging.info(f"Generated customized code (attempt {attempt + 1}): {customized_code}")
            
            # Clean the code if it contains markdown formatting
            if "```python" in customized_code:
                start_index = customized_code.find("```python") + len("```python")
                end_index = customized_code.find("```", start_index)
                customized_code = customized_code[start_index:end_index].strip()
            
            # Remove any imports, variable assignments, or fig.show() calls
            # First, split the code into lines
            lines = customized_code.split('\n')
            cleaned_lines = []
            for line in lines:
                # Skip import statements
                if line.strip().startswith('import'):
                    continue
                # Skip variable assignments
                if '=' in line and not line.strip().startswith('px.'):
                    continue
                # Skip fig.show() calls
                if 'fig.show()' in line:
                    continue
                # Skip standalone fig references
                if line.strip() == 'fig':
                    continue
                cleaned_lines.append(line)
            
            # Join the cleaned lines and strip any extra whitespace
            customized_code = '\n'.join(cleaned_lines).strip()
            
            logging.info(f"Cleaned code after removal of imports and show(): {customized_code}")
            
            # Validate the visualization code
            is_valid, error_message, cleaned_code = validate_visualization_code(customized_code, pd.DataFrame(data))
            logging.info(f"Validation result: is_valid={is_valid}, error={error_message}")
            
            if is_valid:
                # Execute the validated code
                exec_code = f"fig = {cleaned_code}"
                context = {'df': pd.DataFrame(data)}
                exec(exec_code, globals(), context)
                fig = context.get('fig')
                logging.info("Successfully executed and created figure")
                return True, fig, ""
            else:
                logging.warning(f"Customization validation failed (attempt {attempt + 1}/{max_attempts}): {error_message}")
                
                # Add error feedback to the prompt for the next attempt
                prompt += f"\nThe previous attempt failed with error: {error_message}\nPlease fix the code and try again."
                
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in customization (attempt {attempt + 1}/{max_attempts}): {error_message}")
            if attempt == max_attempts - 1:
                return False, {}, error_message
    
    return False, {}, "Failed to generate valid customization after multiple attempts"

if __name__ == '__main__':
    app.run_server(debug=True)