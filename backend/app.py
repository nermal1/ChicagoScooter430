from flask import Flask, render_template, send_from_directory
from dash import Dash, dcc, html
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

server = Flask(
    __name__,
    template_folder=FRONTEND_DIR,      
    static_folder=FRONTEND_DIR         
)

@server.route('/')
def home():
    return render_template('frontpage.html')

@server.route('/frontend/<path:filename>')
def serve_frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# --- Set up Dash app (backend dashboard) ---
app = Dash(
    __name__,
    server=server,
    url_base_pathname='/dashboard/'
)

app.layout = html.Div([
    html.H1("E-Scooter Usage Dashboard"),
    dcc.Dropdown(
        options=[
            {'label': 'Lyft', 'value': 'Lyft'},
            {'label': 'Link', 'value': 'Link'},
            {'label': 'Lime', 'value': 'Lime'},
            {'label': 'Spin', 'value': 'Spin'},
        ],
        placeholder="Select a vendor"
    ),
    dcc.Graph(id='placeholder-graph')
])

# --- Run the server ---
if __name__ == '__main__':
    app.run(debug=True)
