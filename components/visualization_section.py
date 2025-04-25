from dash import html, dcc
import dash_bootstrap_components as dbc
from styles.theme import CARD_STYLE, BUTTON_STYLE, TEXTAREA_STYLE, GRAPH_STYLE

def create_visualization_section():
    return dbc.Card([
        dbc.CardBody([
            html.H4("Visualization", className="card-title"),
            # Graph customization
            html.Div([
                html.Label("Customize Graph:", style={"font-weight": "bold", "margin-bottom": "0.5rem"}),
                dcc.Textarea(
                    id='customize-graph-area',
                    value='',
                    style=TEXTAREA_STYLE,
                    placeholder='Enter your customization request here...'
                ),
                html.Button(
                    'Apply Customization',
                    id='apply-customization-btn',
                    n_clicks=0,
                    style=dict(BUTTON_STYLE, **{"background-color": "#28a745"})
                )
            ]),
            # Graph area
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=dcc.Graph(
                    id='graph-area',
                    style=GRAPH_STYLE
                )
            )
        ])
    ], style=CARD_STYLE) 