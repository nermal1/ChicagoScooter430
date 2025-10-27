import pandas as pd
from sqlalchemy import create_engine
import os

CSV_FILE = os.path.join(os.path.dirname(__file__), "../data/E-Scooter_Trips.csv")
DATABASE_URL = "postgresql://localhost:5432/scooter_db" 
TABLE_NAME = "scooter_trips"

print(f"Loading data from {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

df = df.rename(columns={
    "Trip ID": "trip_id",
    "Start Time": "start_time",
    "End Time": "end_time",
    "Trip Distance": "trip_distance",
    "Trip Duration": "trip_duration",
    "Vendor": "vendor",
    "Start Community Area Number": "start_area_number",
    "End Community Area Number": "end_area_number",
    "Start Community Area Name": "start_area_name",
    "End Community Area Name": "end_area_name",
    "Start Centroid Latitude": "start_centroid_lat",
    "Start Centroid Longitude": "start_centroid_lng",
    "End Centroid Latitude": "end_centroid_lat",
    "End Centroid Longitude": "end_centroid_lng"
})

print(f"Connecting to PostgreSQL at {DATABASE_URL}")
engine = create_engine(DATABASE_URL)

print(f"Writing DataFrame to table '{TABLE_NAME}' in Postgres (this may take a while)...")
df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)

print("Import complete!")
