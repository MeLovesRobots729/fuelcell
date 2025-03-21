import streamlit as st
import numpy as np
import time
import openai
import plotly.graph_objects as go
import os
import re

# Fetch the OpenAI API key directly from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables!")

# Set the API key for OpenAI
openai.api_key = openai_api_key

print("API Key successfully loaded.")  

def ai_optimization(load, hydrogen):
    """Uses AI to optimize hydrogen fuel usage based on load demand."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI optimizing energy distribution for fuel cells. Only return a number indicating the hydrogen needed in kg."},
            {"role": "user", "content": f"Given a load demand of {load} kWh and hydrogen reserves of {hydrogen} kg, optimize fuel distribution."}
        ]
    )
    
    # Get response message
    response_message = response.choices[0].message.content.strip()

    # Extract the first number from the response using regex
    match = re.search(r"\d+(\.\d+)?", response_message)  # Matches integers and decimals
    if match:
        hydrogen_needed = float(match.group())  # Convert to float
        return hydrogen_needed
    else:
        raise ValueError(f"AI response did not contain a valid number: {response_message}")

# Streamlit UI
st.title("Live Fuel Cell AI-Powered Energy Simulation")

# Initial variables
hydrogen_storage = 500  # kg
battery_storage = 200  # kWh

# Initialize charts
fig_load = go.Figure()
fig_hydrogen = go.Figure()

time_steps = 24  # 24-hour simulation
load_demand = np.random.randint(2000, 5000, size=time_steps)  # Random AI data center load
fuel_cell_production = np.zeros(time_steps)
hydrogen_usage = np.zeros(time_steps)
battery_usage = np.zeros(time_steps)

# Simulation Loop
for t in range(time_steps):
    # AI optimizes hydrogen consumption
    hydrogen_needed = ai_optimization(load_demand[t], hydrogen_storage)
    
    if hydrogen_storage >= hydrogen_needed:
        fuel_cell_production[t] = hydrogen_needed * 50  # Assuming 1 kg H2 = 50 kWh
        hydrogen_usage[t] = hydrogen_needed
        hydrogen_storage -= hydrogen_needed
    else:
        fuel_cell_production[t] = 0  # No hydrogen left
        hydrogen_usage[t] = 0

    # Battery backup if fuel cell fails
    if fuel_cell_production[t] < load_demand[t] and battery_storage > 0:
        battery_usage[t] = min(load_demand[t] - fuel_cell_production[t], battery_storage)
        battery_storage -= battery_usage[t]
    
    # Update Graphs
    fig_load.add_trace(go.Scatter(y=load_demand[:t+1], mode='lines', name='Data Center Load Demand (kWh)'))
    fig_load.add_trace(go.Scatter(y=fuel_cell_production[:t+1], mode='lines', name='Fuel Cell Production (kWh)'))
    
    fig_hydrogen.add_trace(go.Scatter(y=hydrogen_usage[:t+1], mode='lines', name='Hydrogen Used (kg)'))
    fig_hydrogen.add_trace(go.Scatter(y=[hydrogen_storage] * (t+1), mode='lines', name='Remaining Hydrogen (kg)'))
    
    # Streamlit display
    st.subheader(f"Hour {t}")
    st.plotly_chart(fig_load, use_container_width=True)
    st.plotly_chart(fig_hydrogen, use_container_width=True)
    
    time.sleep(1)  # Simulate real-time updates
