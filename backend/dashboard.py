import os
import pandas as pd
from sqlalchemy import create_engine, text
from flask import Flask, render_template, send_from_directory, request, jsonify
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

try:
    from ai_agent import answer_question
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False
    print("AI Agent not available. Install crewai and dependencies to enable.")

# Database connection setup
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/scooter_db")
engine = create_engine(DATABASE_URL)

# Flask setup for frontend 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

server = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR
)

@server.route("/")
def home():
    return render_template("frontpage.html")

@server.route("/frontend/<path:filename>")
def serve_frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# AI Agent Route 
@server.route("/ai-ask", methods=["POST"])
def ai_ask():
    if not AI_ENABLED:
        return jsonify({"error": "AI Agent is not enabled. Please install crewai dependencies."})
    
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"})
    
    try:
        result = answer_question(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"AI processing failed: {str(e)}"})

# Dash App Setup
app = Dash(
    __name__,
    server=server,
    url_base_pathname="/dashboard/"
)

COLOR_MAP = {
    "Lyft": "blue",
    "Link": "red",
    "Lime": "green",
    "Spin": "purple"
}

# SQL Query Functions 
def fetch_provider_summary():
    query = text("""
    SELECT vendor,
           COUNT(*) AS total_trips,
           AVG(trip_duration) AS avg_duration,
           AVG(trip_distance) AS avg_distance
    FROM scooter_trips
    GROUP BY vendor
    ORDER BY total_trips DESC;
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def fetch_trips_per_day(vendor=None):
    if vendor:
        query = text("""
        SELECT vendor,
               DATE(start_time) AS date,
               COUNT(*) AS trips
        FROM scooter_trips
        WHERE vendor = :vendor
        GROUP BY vendor, date
        ORDER BY date;
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"vendor": vendor})
    else:
        query = text("""
        SELECT vendor,
               DATE(start_time) AS date,
               COUNT(*) AS trips
        FROM scooter_trips
        GROUP BY vendor, date
        ORDER BY date;
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn)

def fetch_avg_distance_per_hour(vendor=None, min_trip_count=50, max_avg_distance=10000):
    if vendor:
        query = text("""
        SELECT vendor,
               EXTRACT(HOUR FROM start_time::timestamp) AS hour,
               COUNT(*) AS trip_count,
               AVG(trip_distance) AS avg_distance
        FROM scooter_trips
        WHERE vendor = :vendor
        GROUP BY vendor, hour
        HAVING COUNT(*) >= :min_count
        ORDER BY vendor, hour;
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"vendor": vendor, "min_count": min_trip_count})
    else:
        query = text("""
        SELECT vendor,
               EXTRACT(HOUR FROM start_time::timestamp) AS hour,
               COUNT(*) AS trip_count,
               AVG(trip_distance) AS avg_distance
        FROM scooter_trips
        GROUP BY vendor, hour
        HAVING COUNT(*) >= :min_count
        ORDER BY vendor, hour;
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"min_count": min_trip_count})
    
    df = df[df['avg_distance'] <= max_avg_distance]
    return df

def fetch_trip_locations(vendor=None, limit=10000):
    if vendor:
        query = text("""
        SELECT vendor,
               start_centroid_lat AS lat,
               start_centroid_lng AS lon,
               start_area_name AS area_name
        FROM scooter_weather
        WHERE vendor = :vendor
          AND start_centroid_lat IS NOT NULL
          AND start_centroid_lng IS NOT NULL
        LIMIT :limit;
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"vendor": vendor, "limit": limit})
    else:
        query = text("""
        SELECT vendor,
               start_centroid_lat AS lat,
               start_centroid_lng AS lon,
               start_area_name AS area_name
        FROM scooter_weather
        WHERE start_centroid_lat IS NOT NULL
          AND start_centroid_lng IS NOT NULL
        LIMIT :limit;
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})


def ensure_weather_view():
    query = """
    DROP VIEW IF EXISTS scooter_weather CASCADE;
    CREATE VIEW scooter_weather AS
    SELECT
        s.vendor,
        s.trip_id,
        s.start_time::timestamp AS start_time,
        DATE_TRUNC('hour', s.start_time::timestamp) AS weather_hour,
        s.trip_distance,
        s.trip_duration,
        s.start_centroid_lat,
        s.start_centroid_lng,
        s.start_area_name,
        s.end_area_name,
        (w.temperature * 9.0 / 5.0) + 32 AS temperature_f,
        w.temperature AS temperature_c,
        w.precipitation,
        w.humidity,
        w.wind_speed
    FROM scooter_trips s
    LEFT JOIN weather_data w
    ON DATE_TRUNC('hour', s.start_time::timestamp) = w.datetime;
    """
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(query)
        print("Weather view created or replaced successfully.")
    except Exception as e:
        print("Error creating weather view:", e)


def fetch_weather_vs_usage(metric="temperature_f"):
    """Fetch correlation between weather metric and usage"""
    query = text(f"""
    SELECT
        DATE_TRUNC('day', weather_hour) AS date,
        AVG({metric}) AS avg_weather,
        COUNT(*) AS trips
    FROM scooter_weather
    WHERE {metric} IS NOT NULL
    GROUP BY date
    ORDER BY date;
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

try:
    ensure_weather_view()
except Exception as e:
    print(f"Could not create weather view on startup: {e}")

app.layout = html.Div([
    # Modern Header with gradient
    html.Div([
        html.H1("🛴 E-Scooter Analytics Dashboard", 
                style={
                    'margin': '0',
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'color': 'white',
                    'textShadow': '2px 2px 4px rgba(0,0,0,0.2)'
                }),
        html.P("Chicago Mobility Insights", 
               style={
                   'margin': '5px 0 0 0',
                   'fontSize': '1.1rem',
                   'color': 'rgba(255,255,255,0.9)',
                   'fontWeight': '300'
               })
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '40px 20px',
        'textAlign': 'center',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'marginBottom': '30px'
    }),

    # Control Panel Card
    html.Div([
        html.Div([
            html.Div([
                html.Label("🎯 Select Vendor", 
                          style={
                              'fontWeight': '600', 
                              'marginBottom': '8px',
                              'display': 'block',
                              'color': '#374151',
                              'fontSize': '0.95rem'
                          }),
                dcc.Dropdown(
                    id='vendor-dropdown',
                    options=[{'label': 'All Vendors', 'value': 'all'}] + 
                            [{'label': v, 'value': v} for v in ['Lyft', 'Link', 'Lime', 'Spin']],
                    value='all',
                    placeholder="Select a vendor",
                    style={
                        'width': '280px',
                        'fontSize': '0.95rem'
                    }
                ),
            ], style={'marginRight': '30px'}),
            
            html.Div([
                html.Label("📏 Unit System", 
                          style={
                              'fontWeight': '600', 
                              'marginBottom': '8px',
                              'display': 'block',
                              'color': '#374151',
                              'fontSize': '0.95rem'
                          }),
                dcc.RadioItems(
                    id='unit-system',
                    options=[
                        {'label': ' Imperial (mi, °F)', 'value': 'imperial'},
                        {'label': ' Metric (km, °C)', 'value': 'metric'}
                    ],
                    value='imperial',
                    inline=True,
                    style={'fontSize': '0.95rem'},
                    inputStyle={'marginRight': '6px', 'marginLeft': '15px'}
                ),
            ]),
        ], style={
            'display': 'flex', 
            'alignItems': 'flex-start',
            'flexWrap': 'wrap',
            'gap': '20px'
        }),
    ], style={
        'backgroundColor': 'white',
        'padding': '25px',
        'margin': '0 20px 30px 20px',
        'borderRadius': '12px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'border': '1px solid #e5e7eb'
    }),

    # Summary Cards Container
    html.Div(id='summary-cards', style={
        'margin': '0 20px 30px 20px',
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '15px',
        'justifyContent': 'center'
    }),

    # Charts Section
    html.Div([
        html.Div([
            dcc.Graph(id='trips-per-day-chart', 
                     config={'displayModeBar': False},
                     style={'height': '400px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'border': '1px solid #e5e7eb'
        }),

        html.Div([
            dcc.Graph(id='provider-summary-chart',
                     config={'displayModeBar': False},
                     style={'height': '400px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'border': '1px solid #e5e7eb'
        }),

        html.Div([
            dcc.Graph(id='avg-distance-hour-chart',
                     config={'displayModeBar': False},
                     style={'height': '400px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'border': '1px solid #e5e7eb'
        }),

        html.Div([
            dcc.Graph(id='trip-location-map', 
                     config={'scrollZoom': True, 'displayModeBar': False},
                     style={'height': '500px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'border': '1px solid #e5e7eb'
        }),
    ], style={'margin': '0 20px'}),

    # Weather Section Divider
    html.Div([
        html.Div(style={
            'height': '1px',
            'background': 'linear-gradient(90deg, transparent, #d1d5db, transparent)',
            'margin': '40px 0 30px 0'
        }),
        html.H2("🌤️ Weather Impact Analysis", 
                style={
                    'textAlign': 'center',
                    'color': '#1f2937',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '10px'
                }),
        html.P("Explore how weather conditions affect scooter usage patterns",
               style={
                   'textAlign': 'center',
                   'color': '#6b7280',
                   'fontSize': '1.05rem',
                   'marginBottom': '30px'
               })
    ]),

    # Weather Controls
    html.Div([
        html.Label("🌡️ Weather Metric", 
                  style={
                      'fontWeight': '600', 
                      'marginBottom': '8px',
                      'display': 'block',
                      'color': '#374151',
                      'fontSize': '0.95rem'
                  }),
        dcc.Dropdown(
            id='weather-metric-dropdown',
            options=[
                {'label': '🌡️ Temperature', 'value': 'temperature'},
                {'label': '🌧️ Precipitation', 'value': 'precipitation'},
                {'label': '💧 Humidity', 'value': 'humidity'},
                {'label': '💨 Wind Speed', 'value': 'wind_speed'}
            ],
            value='temperature',
            placeholder='Select weather metric',
            style={'width': '280px', 'fontSize': '0.95rem'}
        ),
    ], style={
        'backgroundColor': 'white',
        'padding': '25px',
        'margin': '0 20px 20px 20px',
        'borderRadius': '12px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'border': '1px solid #e5e7eb'
    }),

    # Weather Chart
    html.Div([
        dcc.Graph(id='weather-usage-chart',
                 config={'displayModeBar': False},
                 style={'height': '450px'})
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'margin': '0 20px 20px 20px',
        'borderRadius': '12px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'border': '1px solid #e5e7eb'
    }),

    # Correlation Display
    html.Div(id='correlation-display', style={
        'fontSize': '1.1rem',
        'margin': '0 20px 40px 20px',
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': '#f3f4f6',
        'borderRadius': '12px',
        'fontWeight': '500',
        'color': '#374151',
        'border': '2px solid #e5e7eb'
    }),

    # AI Section Divider
    html.Div([
        html.Div(style={
            'height': '1px',
            'background': 'linear-gradient(90deg, transparent, #d1d5db, transparent)',
            'margin': '40px 0 30px 0'
        }),
        html.H2("🤖 AI Data Assistant", 
                style={
                    'textAlign': 'center',
                    'color': '#1f2937',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '10px'
                }),
        html.P("Ask questions about the data in natural language",
               style={
                   'textAlign': 'center',
                   'color': '#6b7280',
                   'fontSize': '1.05rem',
                   'marginBottom': '30px'
               })
    ]),

    # AI Input Section
    html.Div([
        html.Div([
            dcc.Input(
                id='ai-question-input',
                type='text',
                placeholder='e.g., How many Lyft trips were taken when temperature was above 70°F?',
                style={
                    'width': '100%',
                    'padding': '15px 20px',
                    'fontSize': '1rem',
                    'borderRadius': '10px',
                    'border': '2px solid #e5e7eb',
                    'outline': 'none',
                    'transition': 'border-color 0.2s',
                    'fontFamily': 'inherit'
                }
            ),
        ], style={'flex': '1', 'marginRight': '15px'}),
        
        html.Button(
            '🔍 Ask AI',
            id='ai-ask-button',
            n_clicks=0,
            style={
                'padding': '15px 35px',
                'fontSize': '1rem',
                'fontWeight': '600',
                'backgroundColor': '#10b981',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'cursor': 'pointer',
                'transition': 'all 0.2s',
                'boxShadow': '0 2px 4px rgba(16, 185, 129, 0.3)',
                'whiteSpace': 'nowrap'
            }
        ),
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'backgroundColor': 'white',
        'padding': '25px',
        'margin': '0 20px 20px 20px',
        'borderRadius': '12px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'border': '1px solid #e5e7eb',
        'flexWrap': 'wrap',
        'gap': '15px'
    }),

    # AI Response Container
    dcc.Loading(
        id="loading-ai",
        type="circle",
        color="#667eea",
        children=[
            html.Div(id='ai-response-container', style={'margin': '0 20px 40px 20px'})
        ]
    ),

    dcc.Store(id='ai-response-store'),

    # Footer
    html.Div([
        html.P("E-Scooter Analytics Dashboard | Data-driven mobility insights",
               style={'margin': '0', 'color': '#9ca3af', 'fontSize': '0.9rem'})
    ], style={
        'textAlign': 'center',
        'padding': '30px',
        'backgroundColor': '#f9fafb',
        'borderTop': '1px solid #e5e7eb',
        'marginTop': '40px'
    })

], style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'backgroundColor': '#f9fafb',
    'minHeight': '100vh',
    'paddingBottom': '20px'
})

# helper functions 
def meters_to_miles(meters):
    return meters * 0.000621371

def meters_to_km(meters):
    return meters / 1000

def seconds_to_minutes(seconds):
    return seconds / 60 

@app.callback(
    Output('summary-cards', 'children'),
    Input('vendor-dropdown', 'value'),
    Input('unit-system', 'value')
)
def update_summary_cards(selected_vendor, unit_system):
    df = fetch_provider_summary()
    if selected_vendor and selected_vendor != 'all':
        df = df[df['vendor'] == selected_vendor]
    
    if df.empty:
        return html.Div("No data available", 
                       style={
                           'padding': '40px',
                           'textAlign': 'center',
                           'color': '#6b7280',
                           'fontSize': '1.1rem'
                       })
    
    cards = []
    for _, row in df.iterrows():
        # Distance conversion
        if unit_system == 'imperial':
            distance_val = meters_to_miles(row['avg_distance'])
            distance_unit = 'mi'
        else:
            distance_val = meters_to_km(row['avg_distance'])
            distance_unit = 'km'

        # Duration
        time_val = seconds_to_minutes(row['avg_duration'])

        cards.append(html.Div([
            html.Div([
                html.H3(row['vendor'], 
                       style={
                           'color': COLOR_MAP.get(row['vendor'], '#374151'),
                           'margin': '0 0 15px 0',
                           'fontSize': '1.5rem',
                           'fontWeight': '700'
                       }),
                html.Div([
                    html.Div([
                        html.Div("Total Trips", 
                                style={
                                    'fontSize': '0.85rem',
                                    'color': '#6b7280',
                                    'marginBottom': '5px',
                                    'fontWeight': '500'
                                }),
                        html.Div(f"{int(row['total_trips']):,}", 
                                style={
                                    'fontSize': '1.8rem',
                                    'fontWeight': '700',
                                    'color': '#1f2937'
                                })
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.Div([
                            html.Span("⏱️ ", style={'marginRight': '5px'}),
                            html.Span(f"{time_val:.1f} min", 
                                     style={'fontSize': '1rem', 'color': '#374151'})
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("📍 ", style={'marginRight': '5px'}),
                            html.Span(f"{distance_val:.2f} {distance_unit}", 
                                     style={'fontSize': '1rem', 'color': '#374151'})
                        ])
                    ])
                ])
            ], style={'position': 'relative'})
        ], style={
            'backgroundColor': 'white',
            'border': f'3px solid {COLOR_MAP.get(row["vendor"], "#e5e7eb")}',
            'borderRadius': '12px',
            'padding': '25px',
            'minWidth': '220px',
            'maxWidth': '280px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.05)',
            'transition': 'transform 0.2s, box-shadow 0.2s',
            'cursor': 'default'
        }))

    return cards

@app.callback(
    Output('trips-per-day-chart', 'figure'),
    Input('vendor-dropdown', 'value')
)
def update_trips_per_day_chart(selected_vendor):
    vendor_param = None if selected_vendor == 'all' else selected_vendor
    df = fetch_trips_per_day(vendor_param)
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df['date'] = pd.to_datetime(df['date'])
    
    fig = px.line(df, x='date', y='trips', color='vendor',
                  title="Trips per Day by Vendor",
                  color_discrete_map=COLOR_MAP)
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Trips',
        hovermode='x unified'
    )
    return fig

@app.callback(
    Output('provider-summary-chart', 'figure'),
    Input('vendor-dropdown', 'value'),
    Input('unit-system', 'value')
)
def update_provider_summary_chart(selected_vendor, unit_system):
    df = fetch_provider_summary()
    if selected_vendor and selected_vendor != 'all':
        df = df[df['vendor'] == selected_vendor]
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if unit_system == 'imperial':
        df['avg_distance_converted'] = df['avg_distance'].apply(meters_to_miles)
        distance_unit = 'mi'
    else:
        df['avg_distance_converted'] = df['avg_distance'].apply(meters_to_km)
        distance_unit = 'km'
    
    df['avg_duration_minutes'] = df['avg_duration'].apply(seconds_to_minutes)
    df['avg_duration_formatted'] = df['avg_duration_minutes'].apply(lambda x: f"{x:.2f} min")

    df['avg_distance_formatted'] = df['avg_distance_converted'].apply(lambda x: f"{x:.3f} {distance_unit}")
    
    fig = px.bar(df, x='vendor', y='total_trips',
                 title="Total Trips by Vendor",
                 hover_data={
                     'total_trips': False,
                     'avg_duration_formatted': True,
                     'avg_distance_formatted': True,
                     'vendor': False
                 },
                 color='vendor',
                 color_discrete_map=COLOR_MAP,
                 labels={
                     'avg_duration_formatted': 'Avg Duration',
                     'avg_distance_formatted': 'Avg Distance'
                 })
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>' +
                      'Total Trips: %{y:,}<br>' +
                      'Avg Duration: %{customdata[0]}<br>' +
                      'Avg Distance: %{customdata[1]}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title='Vendor',
        yaxis_title='Total Trips',
        showlegend=False
    )
    return fig

@app.callback(
    Output('avg-distance-hour-chart', 'figure'),
    Input('vendor-dropdown', 'value'),
    Input('unit-system', 'value')
)
def update_avg_distance_hour_chart(selected_vendor, unit_system):
    vendor_param = None if selected_vendor == 'all' else selected_vendor
    df = fetch_avg_distance_per_hour(vendor_param, min_trip_count=50, max_avg_distance=10000)
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if unit_system == 'imperial':
        df['avg_distance'] = df['avg_distance'].apply(lambda x: round(meters_to_miles(x), 2))
        distance_label = 'Average Distance (mi)'
        unit_suffix = 'mi'
    else:
        df['avg_distance'] = df['avg_distance'].apply(lambda x: round(meters_to_km(x), 2))
        distance_label = 'Average Distance (km)'
        unit_suffix = 'km'

    fig = px.line(
        df, x='hour', y='avg_distance', color='vendor',
        title="Average Trip Distance by Hour & Vendor",
        color_discrete_map=COLOR_MAP
    )
    
    fig.update_layout(
        xaxis_title='Hour of Day',
        yaxis_title=distance_label,
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )

    fig.update_traces(
        hovertemplate=f'Hour: %{{x}}<br>Avg Distance: %{{y:.2f}} {unit_suffix}<extra></extra>'
    )

    return fig


@app.callback(
    Output('trip-location-map', 'figure'),
    Input('vendor-dropdown', 'value')
)
def update_location_heatmap(selected_vendor):
    vendor_param = None if selected_vendor == 'all' else selected_vendor
    
    df_all = fetch_trip_locations(vendor=vendor_param, limit=10000)
    
    if df_all.empty:
        return go.Figure().add_annotation(text="No location data available", showarrow=False)
    
    count_query = text("""
        SELECT vendor, start_area_name AS area_name, COUNT(*) as true_count
        FROM scooter_trips 
        WHERE start_area_name IS NOT NULL
        GROUP BY vendor, start_area_name
    """)
        
    with engine.connect() as conn:
        df_counts = pd.read_sql(count_query, conn)

    area_tooltips = {}
    
    if not df_counts.empty:
        grouped = df_counts.groupby('area_name')
        for area, group in grouped:
            group = group.sort_values('true_count', ascending=False)
            
            lines = []
            total_area_trips = 0
            for _, row in group.iterrows():
                lines.append(f"{row['vendor']}: {row['true_count']:,}")
                total_area_trips += row['true_count']

            tooltip_str = f"Total: {total_area_trips:,}<br>----------------<br>" + "<br>".join(lines)
            area_tooltips[area] = tooltip_str

    fig = go.Figure()

    for vendor, color in COLOR_MAP.items():

        df_sub = df_all[df_all['vendor'] == vendor].copy()
        if df_sub.empty:
            continue

        df_sub['hover_text'] = df_sub['area_name'].map(area_tooltips).fillna(f"{vendor}: Data N/A")
        df_sub['hover_area'] = df_sub['area_name']
        
        customdata = df_sub[['hover_area', 'hover_text']].to_numpy()

        fig.add_trace(go.Densitymap(
            lat=df_sub['lat'],
            lon=df_sub['lon'],
            radius=20,
            name=vendor,
            colorscale=[[0, color], [1, color]],
            opacity=0.6,
            customdata=customdata,
            
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" 
                "%{customdata[1]}"         
                "<extra></extra>"             
            ),

            showlegend=True,
            showscale=False,
            legendgroup=vendor
        ))

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=41.8781, lon=-87.6298),
            zoom=11
        ),
        margin={"r":0,"t":30,"l":0,"b":0},
        legend_title="Vendor",
        map_style="open-street-map",
        hovermode='closest'
    )
    
    return fig

@app.callback(
    Output('weather-usage-chart', 'figure'),
    Output('correlation-display', 'children'),
    Input('weather-metric-dropdown', 'value'),
    Input('unit-system', 'value')
)
def update_weather_usage_chart(metric, unit_system):
    if metric == 'temperature':
        db_metric = 'temperature_f' if unit_system == 'imperial' else 'temperature_c'
        metric_label = 'Temperature (°F)' if unit_system == 'imperial' else 'Temperature (°C)'
    else:
        db_metric = metric
        metric_label = metric.replace('_', ' ').title()
        if metric == 'precipitation':
            metric_label = 'Precipitation (mm)'
        elif metric == 'humidity':
            metric_label = 'Humidity (%)'
        elif metric == 'wind_speed':
            metric_label = 'Wind Speed (mph)'
    
    df = fetch_weather_vs_usage(db_metric)

    if df.empty or len(df) < 2:
        fig = go.Figure().add_annotation(
            text="Insufficient weather data",
            showarrow=False,
            font=dict(size=16)
        )
        return fig, "No correlation data available."
    
    if metric == "precipitation":
        bins = [-0.01, 0.0, 2.0, 5.0, float("inf")]
        labels = [
            "No Rain (0mm)",
            "Light Rain (0-2mm)",
            "Moderate Rain (2-5mm)",
            "Heavy Rain (>5mm)"
        ]

        query = text("""
            SELECT precipitation, trip_distance, trip_duration
            FROM scooter_weather
            WHERE precipitation IS NOT NULL
        """)
        with engine.connect() as conn:
            trip_df = pd.read_sql(query, conn)

        duration_label = "Avg Duration (min)" 

        if unit_system == "imperial":
            trip_df["trip_distance"] *= 0.000621371  
            distance_label = "Avg Distance (mi)"
            unit_suffix = "mi" 
        else:
            trip_df["trip_distance"] /= 1000  
            distance_label = "Avg Distance (km)"
            unit_suffix = "km" 


        trip_df["trip_duration"] /= 60  

        trip_df["rain_bucket"] = pd.cut(trip_df["precipitation"], bins=bins, labels=labels)

        bucket_stats = trip_df.groupby("rain_bucket", observed=False).agg(
            avg_distance=("trip_distance", "mean"),
            avg_duration=("trip_duration", "mean"),
            total_trips=("trip_distance", "count")
        ).reset_index()

        # No melting needed for dual-axis manual construction
        
        fig = go.Figure()

        # Trace 1: Distance (Left Y, Slot 1)
        fig.add_trace(go.Bar(
            x=bucket_stats['rain_bucket'],
            y=bucket_stats['avg_distance'],
            name=distance_label,
            yaxis='y1',
            offsetgroup=1,
            marker_color='#636EFA',  # Plotly Blue
            hovertemplate=f"<b>%{{x}}</b><br>{distance_label}: %{{y:.2f}} {unit_suffix}<extra></extra>"
        ))

        # Trace 2: Duration (Left Y, Slot 2)
        fig.add_trace(go.Bar(
            x=bucket_stats['rain_bucket'],
            y=bucket_stats['avg_duration'],
            name=duration_label,
            yaxis='y1',
            offsetgroup=2,
            marker_color='#EF553B',  # Plotly Red
            hovertemplate=f"<b>%{{x}}</b><br>{duration_label}: %{{y:.2f}} min<extra></extra>"
        ))

        # Trace 3: Total Trips (Right Y, Slot 3)
        fig.add_trace(go.Bar(
            x=bucket_stats['rain_bucket'],
            y=bucket_stats['total_trips'],
            name='Total Trips',
            yaxis='y2',
            offsetgroup=3,
            marker_color='#00CC96', # Plotly Green
            hovertemplate="<b>%{x}</b><br>Total Trips: %{y:,}<extra></extra>"
        ))

        fig.update_layout(
            title="Scooter Behavior by Rainfall Category",
            xaxis=dict(title="Rain Category"),
            yaxis=dict(
                title="Average Distance / Duration",
                side='left'
            ),
            yaxis2=dict(
                title="Total Trips",
                side='right',
                overlaying='y',
                showgrid=False
            ),
            barmode='group',
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
            height=500
        )

        return fig, "Showing averages (left axis) and total trip volume (right axis) per rainfall bucket."

    corr = df["avg_weather"].corr(df["trips"])

    fig = px.scatter(
        df,
        x="avg_weather",
        y="trips",
        title=f"Impact of {metric_label} on Scooter Usage",
        labels={"avg_weather": metric_label, "trips": "Trips per Day"},
    )

    fig.update_traces(marker=dict(size=8, opacity=0.6, color='blue'))
    fig.update_layout(height=500)

    correlation_text = f"Correlation between {metric_label.lower()} and trips: {corr:.3f}"
    if abs(corr) > 0.7:
        correlation_text += " (Strong)"
    elif abs(corr) > 0.4:
        correlation_text += " (Moderate)"
    else:
        correlation_text += " (Weak)"

    return fig, correlation_text

@app.callback(
    Output('ai-response-container', 'children'),
    Input('ai-ask-button', 'n_clicks'),
    State('ai-question-input', 'value'),
    prevent_initial_call=True
)
def handle_ai_question(n_clicks, question):
    if not question:
        return html.Div(
            "⚠️ Please enter a question!",
            style={
                'padding': '20px',
                'backgroundColor': '#fff3cd',
                'border': '1px solid #ffc107',
                'borderRadius': '5px',
                'color': '#856404'
            }
        )
    
    if not AI_ENABLED:
        return html.Div([
            html.H4("⚠️ AI Agent Not Available", style={'color': '#dc3545'}),
            html.P("The AI agent requires CrewAI and its dependencies to be installed."),
            html.P("Install with: pip install crewai crewai-tools")
        ], style={
            'padding': '20px',
            'backgroundColor': '#f8d7da',
            'border': '1px solid #dc3545',
            'borderRadius': '5px'
        })
    
    # Make request to Flask endpoint
    import requests
    try:
        response = requests.post(
            'http://localhost:8050/ai-ask',
            json={'question': question},
            timeout=60
        )
        result = response.json()
        
        if 'error' in result:
            return html.Div([
                html.H4("Error", style={'color': '#dc3545'}),
                html.P(result['error'])
            ], style={
                'padding': '20px',
                'backgroundColor': '#f8d7da',
                'border': '1px solid #dc3545',
                'borderRadius': '5px'
            })
        
        # Display the answer
        return html.Div([
            html.H4("Question:", style={'color': '#007bff', 'marginBottom': '10px'}),
            html.P(question, style={'fontStyle': 'italic', 'marginBottom': '20px'}),
            
            html.H4("Answer:", style={'color': '#28a745', 'marginBottom': '10px'}),
            html.P(result.get('answer', 'No answer provided'), 
                   style={'fontSize': '16px', 'lineHeight': '1.6', 'marginBottom': '20px'}),
            
            html.Details([
                html.Summary("View SQL Query", 
                            style={'cursor': 'pointer', 'color': '#6c757d', 'fontWeight': 'bold'}),
                html.Pre(result.get('sql', 'No SQL generated'), 
                        style={
                            'backgroundColor': '#f8f9fa',
                            'padding': '15px',
                            'borderRadius': '5px',
                            'overflow': 'auto',
                            'marginTop': '10px'
                        })
            ], style={'marginTop': '15px'})
        ], style={
            'padding': '25px',
            'backgroundColor': '#d4edda',
            'border': '2px solid #28a745',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
        
    except requests.exceptions.Timeout:
        return html.Div([
            html.H4("Timeout", style={'color': '#dc3545'}),
            html.P("The AI agent took too long to respond. Please try a simpler question.")
        ], style={
            'padding': '20px',
            'backgroundColor': '#f8d7da',
            'border': '1px solid #dc3545',
            'borderRadius': '5px'
        })
    except Exception as e:
        return html.Div([
            html.H4("Error", style={'color': '#dc3545'}),
            html.P(f"Failed to connect to AI agent: {str(e)}")
        ], style={
            'padding': '20px',
            'backgroundColor': '#f8d7da',
            'border': '1px solid #dc3545',
            'borderRadius': '5px'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)