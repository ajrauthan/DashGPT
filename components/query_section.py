from dash import html, dcc
import dash_bootstrap_components as dbc
from styles.theme import CARD_STYLE, BUTTON_STYLE, TEXTAREA_STYLE

def create_query_section():
    return dbc.Card([
        dbc.CardBody([
            html.H4("Ask your question", className="card-title"),
            dcc.Textarea(
                id='question-area',
                value='',
                style=TEXTAREA_STYLE,
                placeholder='Enter your question here...'
            ),
            html.Button('Submit', id='submit-btn', n_clicks=0, style=BUTTON_STYLE)
        ])
    ], style=CARD_STYLE) 