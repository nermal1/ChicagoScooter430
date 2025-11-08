import pandas as pd
from sqlalchemy import create_engine
import os

CSV_FILE = os.path.join(os.path.dirname(__file__), "../data/weatherData.csv")
DATABASE_URL = "postgresql://localhost:5432/scooter_db" 
TABLE_NAME = "weather_data"

print(f"Loading weather data from {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

df = df.rename(columns={
    "YEAR": "year",
    "MO": "month",
    "DY": "day",
    "HR": "hour",
    "TEMP": "temperature",
    "PRCP": "precipitation",
    "HMDT": "humidity",
    "WND_SPD": "wind_speed",
    "ATM_PRESS": "pressure",
    "REF": "reference"
})

df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

print(f"Connecting to PostgreSQL at {DATABASE_URL}")
engine = create_engine(DATABASE_URL)

print(f"Writing DataFrame to table '{TABLE_NAME}' in Postgres (this may take a while)...")
df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

print("Weather import complete!")
