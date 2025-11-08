import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/scooter_db")
engine = create_engine(DATABASE_URL)

# LLM setupt using Groq

try:
    from crewai import Agent, Crew, Task, Process, LLM
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("CrewAI not installed. Install with: pip install crewai crewai-tools")
    def tool(func):
        return func

# Determine which LLM to use

llm = None
if CREWAI_AVAILABLE and os.getenv("GROQ_API_KEY"):
    try:
        llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        print("Using Groq Llama 3.3 70B")
    except Exception as e:
        print(f"Failed to initialize Groq LLM: {e}")
else:
    print("⚠️ GROQ_API_KEY not found in environment")

@tool
def execute_sql_query(query: str) -> str:
    """Execute SELECT queries safely against the scooter database."""
    query = query.strip()
    
    if query.startswith('```'):
        query = '\n'.join([line for line in query.split('\n') if not line.startswith('```')]).strip()
 
    if query.lower().startswith('sql'):
        query = query[3:].strip()

    if not query.upper().startswith('SELECT'):
        return "Error: Only SELECT queries are allowed."

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if df.empty:
            return "Query executed successfully but returned no results."
        return df.to_string(index=False, max_rows=100)
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def get_database_schema() -> str:
    """Return schema info for all relevant tables."""
    return """
Available Tables and Columns:

1. scooter_trips:
   - trip_id: Unique trip identifier
   - vendor: Scooter provider (Lyft, Link, Lime, Spin)
   - start_time: Trip start timestamp
   - trip_duration: Duration in seconds
   - trip_distance: Distance in meters
   - start_centroid_lat: Starting latitude
   - start_centroid_lng: Starting longitude
   - start_area_name: Community area name of the trip start
   - end_area_name: Community area name of the trip end

2. weather_data:
   - datetime: Timestamp (hourly)
   - temperature: Temperature in Celsius
   - precipitation: Precipitation in mm
   - humidity: Humidity percentage
   - wind_speed: Wind speed in mph

3. scooter_weather (VIEW - combines scooter_trips + weather_data):
   - vendor: Scooter provider
   - trip_id: Unique trip identifier
   - start_time: Trip start timestamp
   - weather_hour: Hour of weather reading
   - start_area_name: Community area name of the trip start
   - end_area_name: Community area name of the trip end
   - trip_distance: Distance in meters
   - trip_duration: Duration in seconds
   - temperature_f: Temperature in Fahrenheit
   - temperature_c: Temperature in Celsius
   - precipitation: Precipitation in mm
   - humidity: Humidity percentage
   - wind_speed: Wind speed in mph

Tips:
- Use scooter_weather view for queries combining trips & weather.
- Vendor names: 'Lyft', 'Link', 'Lime', 'Spin' (case-sensitive)
- For location questions (e.g., 'where', 'place', 'area'), use 'start_area_name' or end_area_name.
- Use DATE_TRUNC or DATE for date-based queries.
- For temperature in Fahrenheit, use temperature_f column
"""

# Create agents and define roles
def create_agents():
    """Create the AI agents for the Crew using the selected LLM."""
    if not CREWAI_AVAILABLE or not llm:
        raise ImportError("CrewAI is not installed or LLM not configured.")

    sql_agent = Agent(
        role="SQL Query Expert",
        goal=(
            "Generate accurate, safe SQL SELECT queries based on natural language questions. "
            "Use the scooter_weather view for queries involving weather. "
            "Output ONLY the SQL query without any markdown formatting or explanations."
        ),
        backstory=(
            "Expert PostgreSQL developer familiar with e-scooter trip data and weather analysis. "
            "You write clean, efficient queries that answer questions precisely."
        ),
        tools=[get_database_schema],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    executor_agent = Agent(
        role="Database Query Executor",
        goal="Execute SQL queries safely and return clean, formatted results.",
        backstory=(
            "Careful database administrator who executes queries and formats output clearly. "
            "You handle errors gracefully."
        ),
        tools=[execute_sql_query],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    analyst_agent = Agent(
        role="Data Analyst",
        goal=
            "Provide a consice, user-friendly answer to the users question"
            "based on the SQL data. your goal is supposed to be brief and get right to the point"
            "Do not write long, repetitive reports",
        backstory=(
            "You are a friendly data assistant. You give the main answer first"
            "Then one or two simple sentence of explanation"
            "You avoid jargon and unecessary statistics"
            "You get straight to the point"
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return sql_agent, executor_agent, analyst_agent

def create_crew(question: str, units: str):
    """Create the multi-agent Crew pipeline for a specific question."""
    sql_agent, executor_agent, analyst_agent = create_agents()

    if units == 'imperial':
        unit_pref_sql = (
            "1. For temperature, use the 'temperature_f' column.\n"
            "2. For distance, convert 'trip_distance' (in meters) to miles. "
            "Tip: Use `(trip_distance * 0.000621371) AS distance_miles`."
        )
        unit_pref_analysis = (
            "Report all temperatures in Fahrenheit (°F) and distances in miles."
        )
    else: 
        unit_pref_sql = (
            "1. For temperature, use the 'temperature_c' column.\n"
            "2. For distance, convert 'trip_distance' (in meters) to kilometers. "
            "Tip: Use `(trip_distance / 1000) AS distance_km`."
        )
        unit_pref_analysis = (
            "Report all temperatures in Celsius (°C) and distances in kilometers (km)."
        )

    # Task 1: Generate SQL
    task1 = Task(
        description=(
            f"User question: {question}\n\n"
            f"**CRITICAL: The user's system preference is for these units:** {unit_pref_sql}\n\n"
            "Steps:\n"
            "1. First, call the 'get_database_schema' tool.\n"
            "2. Analyze the user question.\n"
            "3. **Always generate the SQL query using the units specified in the 'CRITICAL' instruction above.**\n"
            "4. **If the user's question includes a *different* unit** (e.g., they ask for 'miles' but the system unit is 'km'), "
               "**you must still use the 'CRITICAL' system units** for the SQL query.\n"
            "5. **For location questions** (e.g., 'where', 'place'), you MUST use 'start_area_name' or 'end_area_name'.\n"
            "6. Generate the PostgreSQL SELECT query.\n"
            "7. **IMPORTANT: Alias the unit-dependent columns** to make it clear what unit you used (e.g., `AS avg_temp_c`, `AS total_distance_km`).\n"
            "8. Output ONLY the SQL query as plain text."
        ),
        expected_output="A PostgreSQL SELECT query as plain text, using the system's required units and aliasing the output columns (e.g., AS avg_temp_c).",
        agent=sql_agent
    )

    # Task 2: Execute SQL
    task2 = Task(
        description=(
            "Take the SQL query from the previous task and execute it using the "
            "'execute_sql_query' tool. Return the complete results."
        ),
        expected_output="Query results as a formatted table",
        agent=executor_agent,
        context=[task1]
    )
    # Task 3: Analyze & Summarize
    task3 = Task(
        description=(
            f"Original question: {question}\n\n"
            f"**CRITICAL: You MUST report units using this rule:** {unit_pref_analysis}\n\n"
            "Analyze the SQL results (from task 2) using the SQL query (from task 1) for context.\n"
            "Follow this structure **exactly**:\n"
            "1. **Direct Answer:** State the main number or finding directly in one sentence.\n"
            "2. **Context/Meaning:** Add one or two simple sentences that put the answer into context.\n\n"
            
            "**ULTRA-IMPORTANT RULES (DO NOT IGNORE):**\n"
            "1. **YOU MUST USE THE UNITS from the 'CRITICAL' instruction.** "
               "(e.g., if it says report in °C, you MUST use °C).\n"
            "2. **The user's question might use different numbers** (e.g., 'below 0'). "
               "**DO NOT** repeat the user's number. State the answer using the units "
               "from the SQL query (e.g., '...when the temperature is below 0°C...').\n"
            "3. **DO NOT CONVERT** the numbers you get from the SQL query. Report them *as-is*.\n"
            "4. **DO NOT** explain *how* the query was made (e.g., 'The query filtered for...').\n"
            "5. **DO NOT** use meta-commentary (e.g., 'This result answers the question...')."
        ),
        expected_output="A short, natural language answer (2-3 sentences max) that gives the main "
                        "answer and a single, simple insight about it, **using the correct metric or imperial units as requested AND NOT CONVERTING THEM.**",
        agent=analyst_agent,
        context=[task1, task2]
    )

    crew = Crew(
        agents=[sql_agent, executor_agent, analyst_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=False
    )

    return crew

def answer_question(question: str, units: str = 'imperial'):
    """Answer a user question using the Crew pipeline with Groq."""
    if not question:
        return {"error": "No question provided"}
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI not installed. Install with: pip install crewai"}
    if not llm:
        return {"error": "No LLM configured. Set GROQ_API_KEY in your .env file"}

    try:
        crew = create_crew(question, units)
        result = crew.kickoff()

        sql_query = "No SQL generated"
        data_result = "No data"
        final_answer = "No answer generated"

        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) > 0 and result.tasks_output[0]:
                sql_query = str(result.tasks_output[0].raw)
            if len(result.tasks_output) > 1 and result.tasks_output[1]:
                data_result = str(result.tasks_output[1].raw)[:1000]
            if len(result.tasks_output) > 2 and result.tasks_output[2]:
                final_answer = str(result.tasks_output[2].raw)

        if final_answer == "No answer generated" and hasattr(result, 'raw'):
            final_answer = str(result.raw)

        return {
            "sql": sql_query,
            "data": data_result,
            "answer": final_answer
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in answer_question:\n{error_trace}")
        return {
            "error": f"AI Crew execution failed: {str(e)}",
            "sql": "",
            "data": "",
            "answer": ""
        }

