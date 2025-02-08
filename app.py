import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("predictive_maintenance_model.pkl")

st.title("Equipment Health Monitoring Dashboard")

st.sidebar.header("Input Equipment Data")

# Explanation of operation modes
st.sidebar.markdown("""
### **Operation Modes**
- **H (High Load)**: Equipment is under heavy workload/stress.
- **M (Medium Load)**: Normal operation at moderate workload.
- **L (Low Load)**: Light workload or idle state.
""")

# One-hot encoded categorical variables (H, M, L)
operation_mode = st.sidebar.radio("Select Operation Mode:", ["H (High Load)", "M (Medium Load)", "L (Low Load)"])
H, M, L = 0, 0, 0  # Default values
if operation_mode.startswith("H"):
    H = 1
elif operation_mode.startswith("M"):
    M = 1
else:
    L = 1

# Numeric features
process_temp = st.sidebar.number_input("Process Temperature (K)", min_value=250.0, max_value=400.0, step=0.1, value=300.0)
air_temp = st.sidebar.number_input("Air Temperature (K)", min_value=250.0, max_value=400.0, step=0.1, value=295.0)
tool_wear = st.sidebar.number_input("Tool Wear (min)", min_value=0, max_value=500, step=1, value=10)
rotational_speed = st.sidebar.number_input("Rotational Speed (RPM)", min_value=100.0, max_value=5000.0, step=10.0, value=1500.0)
torque = st.sidebar.number_input("Torque (Nm)", min_value=0.0, max_value=500.0, step=0.1, value=50.0)

# Prepare input data with 8 features
# Add a dummy column to match the scaler's expected input shape
dummy_target = 0  # Placeholder value (won't be used)
input_data = np.array([[H, M, L, process_temp, air_temp, tool_wear, rotational_speed, torque, dummy_target]])
scaled_data = scaler.transform(input_data)


# Predict equipment health
prediction = model.predict(scaled_data)[0]
prediction=int(prediction.flatten()[0])

st.subheader("🔍 Prediction Result")
if prediction == 0:
    st.markdown("<h2 style='color: green;'>🟢 Equipment is Healthy</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='color: red;'>🔴 Needs Maintenance</h2>", unsafe_allow_html=True)

health_score = np.random.randint(60, 100)  # Simulated health score

st.subheader("📊 Equipment Health Score")
st.progress(health_score / 100)  # Convert to 0-1 scale
st.write(f"🔹 **Current Health Score:** {health_score}/100")

# Display historical trend (dummy data for now)
st.subheader("📈 Equipment Health Trends")

# Generate dummy trend data
dummy_data = pd.DataFrame({
    "Time": pd.date_range(start="2024-01-01", periods=10, freq="D"),
    "Health Score": np.random.randint(50, 100, size=10)
})

# Create and customize the trend line chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dummy_data["Time"], dummy_data["Health Score"], marker='o', linestyle='-', color='blue', label="Health Score")
ax.set_title("Equipment Health Over Time", fontsize=14)
ax.set_xlabel("Date")
ax.tick_params(axis='x', rotation=45)
ax.set_ylabel("Health Score")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

st.pyplot(fig)

# Organize feature metrics in two columns
st.subheader("⚡ Equipment Parameters Overview")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="🌡️ Process Temperature (K)", value=f"{process_temp} K")
    st.metric(label="🌡️ Air Temperature (K)", value=f"{air_temp} K")

with col2:
    st.metric(label="🛠️ Tool Wear (min)", value=f"{tool_wear} min")
    st.metric(label="⚡ Torque (Nm)", value=f"{torque} Nm")
