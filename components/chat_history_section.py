from dash import html
import dash_bootstrap_components as dbc
from styles.theme import CARD_STYLE, BUTTON_STYLE, CHAT_HISTORY_STYLE

def create_chat_history_section():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Button(
                    "View Chat History",
                    id="toggle-chat-history",
                    n_clicks=0,
                    style=BUTTON_STYLE
                ),
                html.Button(
                    "Clear Chat History",
                    id="clear-chat-history",
                    n_clicks=0,
                    style=dict(BUTTON_STYLE, **{"background-color": "#dc3545"})
                )
            ]),
            dbc.Collapse(
                html.Div(
                    id='chat-history-area',
                    style=CHAT_HISTORY_STYLE
                ),
                id='collapse-chat-history',
                is_open=False,
            )
        ])
    ], style=CARD_STYLE) 