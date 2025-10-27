import os
import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, render_template, send_from_directory
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# --- Database connection setup ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/scooter_db")
engine = create_engine(DATABASE_URL)

# --- Flask setup for frontend ---
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

def fetch_provider_summary():
    query = """
    SELECT vendor,
           COUNT(*) AS total_trips,
           AVG(trip_duration) AS avg_duration,
           AVG(trip_distance) AS avg_distance
    FROM scooter_trips
    GROUP BY vendor
    ORDER BY total_trips DESC;
    """
    return pd.read_sql(query, engine)

def fetch_trips_per_day(vendor=None):
    vendor_filter = f"WHERE vendor = '{vendor}'" if vendor else ""
    query = f"""
    SELECT vendor,
           DATE(start_time) AS date,
           COUNT(*) AS trips
    FROM scooter_trips
    {vendor_filter}
    GROUP BY vendor, date
    ORDER BY date;
    """
    return pd.read_sql(query, engine)

def fetch_avg_distance_per_hour(vendor=None, min_trip_count=50, max_avg_distance=10000):
    vendor_filter = f"WHERE vendor = '{vendor}'" if vendor else ""
    query = f"""
    SELECT vendor,
           date_part('hour', start_time::timestamp) AS hour,
           COUNT(*) AS trip_count,
           AVG(trip_distance) AS avg_distance
    FROM scooter_trips
    {vendor_filter}
    GROUP BY vendor, hour
    HAVING COUNT(*) >= {min_trip_count}
    ORDER BY vendor, hour;
    """
    df = pd.read_sql(query, engine)
    df = df[df['avg_distance'] <= max_avg_distance]
    return df

def fetch_trip_locations(vendor=None, limit=10000):
    vendor_filter = f"WHERE vendor = '{vendor}'" if vendor else ""
    query = f"""
    SELECT vendor,
           start_centroid_lat AS lat,
           start_centroid_lng AS lon
    FROM scooter_trips
    {vendor_filter}
    WHERE start_centroid_lat IS NOT NULL
      AND start_centroid_lng IS NOT NULL
    LIMIT {limit};
    """
    return pd.read_sql(query, engine)

app.layout = html.Div([
    html.H1("E-Scooter Usage Dashboard (Chicago)"),
    dcc.Dropdown(
        id='vendor-dropdown',
        options=[{'label': v, 'value': v} for v in fetch_provider_summary()['vendor'].unique()],
        value=None,
        placeholder="Select a vendor (or all)"
    ),
    html.Div(id='summary-cards'),
    dcc.Graph(id='trips-per-day-chart'),
    dcc.Graph(id='provider-summary-chart'),
    dcc.Graph(id='avg-distance-hour-chart'),
    dcc.Graph(id='trip-location-map')
])

@app.callback(
    Output('summary-cards', 'children'),
    Input('vendor-dropdown', 'value')
)
def update_summary_cards(selected_vendor):
    df = fetch_provider_summary()
    if selected_vendor:
        df = df[df['vendor'] == selected_vendor]
    cards = []
    for _, row in df.iterrows():
        cards.append(html.Div([
            html.H3(row['vendor']),
            html.P(f"Total Trips: {int(row['total_trips'])}"),
            html.P(f"Avg Duration: {row['avg_duration']:.1f} s"),
            html.P(f"Avg Distance: {row['avg_distance']:.1f} m"),
        ], style={'border':'1px solid #ccc','padding':'10px','margin':'5px','display':'inline-block'}))
    return cards

@app.callback(
    Output('trips-per-day-chart', 'figure'),
    Input('vendor-dropdown', 'value')
)
def update_trips_per_day_chart(selected_vendor):
    df = fetch_trips_per_day(selected_vendor)
    fig = px.line(df, x='date', y='trips', color='vendor',
                  title="Trips per Day by Vendor",
                  color_discrete_map=COLOR_MAP)
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Trips')
    return fig

@app.callback(
    Output('provider-summary-chart', 'figure'),
    Input('vendor-dropdown', 'value')
)
def update_provider_summary_chart(selected_vendor):
    df = fetch_provider_summary()
    if selected_vendor:
        df = df[df['vendor'] == selected_vendor]
    fig = px.bar(df, x='vendor', y='total_trips',
                 title="Total Trips by Vendor",
                 hover_data=['avg_duration','avg_distance'],
                 color='vendor',
                 color_discrete_map=COLOR_MAP)
    fig.update_layout(xaxis_title='Vendor', yaxis_title='Total Trips')
    return fig

@app.callback(
    Output('avg-distance-hour-chart', 'figure'),
    Input('vendor-dropdown', 'value')
)
def update_avg_distance_hour_chart(selected_vendor):
    df = fetch_avg_distance_per_hour(selected_vendor,
                                      min_trip_count=50,
                                      max_avg_distance=10000)
    fig = px.line(df, x='hour', y='avg_distance', color='vendor',
                  title="Average Trip Distance by Hour & Vendor",
                  color_discrete_map=COLOR_MAP)
    fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Average Distance (m)')
    return fig

@app.callback(
    Output('trip-location-map', 'figure'),
    Input('vendor-dropdown', 'value')
)
def update_location_heatmap(selected_vendor):
    df_all = fetch_trip_locations(vendor=None, limit=10000)
    fig = go.Figure()
    for vendor, color in COLOR_MAP.items():
        if selected_vendor and vendor != selected_vendor:
            continue
        df_sub = df_all[df_all['vendor'] == vendor]
        if df_sub.empty:
            continue
        fig.add_trace(go.Densitymap(
            lat=df_sub['lat'].tolist(),
            lon=df_sub['lon'].tolist(),
            radius=20,
            name=vendor,
            colorscale=[[0, color], [1, color]],
            opacity=0.6,
            hovertemplate=f"{vendor}<br>Lat: %{{lat}}<br>Lon: %{{lon}}<extra></extra>"
        ))

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=41.8781, lon=-87.6298),
            zoom=11
        ),
        margin={"r":0,"t":30,"l":0,"b":0},
        legend_title="Vendor"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
