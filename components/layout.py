from dash import html, dcc
import dash_bootstrap_components as dbc
from styles.theme import CONTENT_STYLE, CARD_STYLE, TEXTAREA_STYLE, BUTTON_STYLE, GRAPH_STYLE
from components.query_section import create_query_section
from components.summary_section import create_summary_section
from components.chat_history_section import create_chat_history_section
from components.visualization_section import create_visualization_section

def create_layout():
    return html.Div([
        # Header
        html.Div([
            html.H1("ChatGPT - A Data Query Chatbot and Visualization Tool", 
                    style={"text-align": "center", "color": "#2c3e50", "margin-bottom": "2rem"})
        ]),
        
        # Main content container
        html.Div([
            # Left column - Query and Chat
            dbc.Row([
                dbc.Col([
                    create_query_section(),
                    create_summary_section(),
                    create_chat_history_section()
                ], width=5),
                
                # Right column - Visualization
                dbc.Col([
                    create_visualization_section()
                ], width=7)
            ])
        ], style=CONTENT_STYLE),
        
        # Store components
        dcc.Store(id='query-result-store'),
        dcc.Store(id='graph-code-store')
    ]) 