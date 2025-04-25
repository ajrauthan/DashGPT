from dash import html, dcc
import dash_bootstrap_components as dbc
from styles.theme import CARD_STYLE, BUTTON_STYLE

def create_summary_section():
    return dbc.Card([
        dbc.CardBody([
            html.H4("Summary", className="card-title"),
            dcc.Loading(
                id="loading-summary",
                type="circle",
                children=html.Div(id='summary-area', style={'min-height': '50px'})
            ),
            html.Button(
                'Rephrase Summary',
                id='refresh-summary-btn',
                n_clicks=0,
                style=dict(BUTTON_STYLE, **{"background-color": "#6c757d", "margin-top": "10px"})
            )
        ])
    ], style=CARD_STYLE) 