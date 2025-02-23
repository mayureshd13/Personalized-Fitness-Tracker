import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import percentileofscore

# Set Page Configuration
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🏋️", layout="wide")

# Apply Dark Theme with Custom CSS
st.markdown("""
    <style>
        body { background-color: #1e1e1e; color: white; }
        .big-font { font-size: 24px !important; font-weight: bold; color: #ff4b4b; }
        .metric { font-size: 20px !important; font-weight: bold; color: #00ffcc; }
        .success { font-size: 18px !important; color: #66ff66; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🏋️ Personal Fitness Tracker</p>', unsafe_allow_html=True)
st.write("A smart app to track your fitness, sleep, and overall health! 🚀")

# Load Datasets
@st.cache_data
def load_data():
    calories_df = pd.read_csv("calories_dataset.csv", encoding="ISO-8859-1")
    fitness_df = pd.read_csv("fitness_dataset.csv", encoding="ISO-8859-1")
    sleep_df = pd.read_csv("sleep_health_dataset.csv", encoding="ISO-8859-1")
    diet_df = pd.read_csv("All_Diets.csv", encoding="ISO-8859-1")

    if "Activity, Exercise" in calories_df.columns:
        calories_df.rename(columns={"Activity, Exercise": "Activity"}, inplace=True)

    return calories_df, fitness_df, sleep_df, diet_df

calories_df, fitness_df, sleep_df, diet_df = load_data()

# Sidebar - User Inputs
st.sidebar.header("👤 User Input")
with st.sidebar.expander("🏋️ Fitness Details", expanded=True):
    weight = st.slider("Enter your weight (kg)", 40, 150, 70)
    height = st.slider("Enter your height (cm)", 140, 200, 170)
    age = st.slider("Enter your age", 18, 80, 30)
    gender = st.radio("Select Gender", ["Male", "Female"])
    heart_rate = st.slider("Heart Rate During Exercise (bpm)", 60, 200, 120)
    body_temp = st.slider("Body Temperature During Exercise (°C)", 35.0, 40.0, 37.0)
    duration = st.slider("Exercise Duration (mins)", 10, 120, 30)

with st.sidebar.expander("📌 Activity Details", expanded=False):
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
    if "Activity" in calories_df.columns:
        exercise_type = st.selectbox("Exercise Type", calories_df["Activity"].unique())
    else:
        st.error("'Activity' column not found in Calories dataset.")

# Sleep Analysis Inputs
with st.sidebar.expander("🌙 Sleep Details", expanded=False):
    sleep_duration = st.slider("Sleep Duration (hrs)", 3, 12, 7)
    sleep_quality = st.select_slider("Sleep Quality", ["Poor", "Average", "Good", "Excellent"], value="Good")
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

# Calculate BMI
bmi = round(weight / ((height / 100) ** 2), 2)

# Calories Burned Calculation
activity_METs = {"Sedentary": 1.2, "Light": 2.5, "Moderate": 5.0, "Active": 7.0, "Very Active": 9.0}
calories_burned = round(activity_METs[activity_level] * weight * (duration / 60), 2)

# Display Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Your BMI", f"{bmi}")
with col2:
    st.metric("🔥 Calories Burned", f"{calories_burned} kcal")
with col3:
    st.metric("😴 Sleep Duration", f"{sleep_duration} hrs")

# Fitness Score Calculation (Responsive to Inputs)
def calculate_fitness_score(weight, height, age, gender, heart_rate, body_temp, duration, bmi):
    # Constants for ideal values (you can adjust this as needed)
    ideal_bmi_range = (18.5, 24.9)
    ideal_heart_rate = (90, 130)  # Target heart rate during exercise
    ideal_temp = 37.0  # Ideal body temperature during exercise (Celsius)
    ideal_duration = 45  # Duration of exercise in minutes
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Score for BMI
    if ideal_bmi_range[0] <= bmi <= ideal_bmi_range[1]:
        bmi_score = 25  # Full score for ideal BMI
    else:
        bmi_score = max(0, 25 - abs(bmi - 22))  # Scale the BMI score between 0-25
    
    # Score for Heart Rate
    if ideal_heart_rate[0] <= heart_rate <= ideal_heart_rate[1]:
        heart_rate_score = 20  # Full score for ideal heart rate
    else:
        heart_rate_score = max(0, 20 - abs(heart_rate - 110))  # Adjust as per ideal HR
    
    # Score for Body Temperature
    if body_temp == ideal_temp:
        temp_score = 20  # Full score for ideal body temperature
    else:
        temp_score = max(0, 20 - abs(body_temp - 37.0))  # Scale based on ideal temperature
    
    # Score for Exercise Duration
    if duration >= ideal_duration:
        duration_score = 25  # Full score for exercise duration above 45 minutes
    else:
        duration_score = duration / ideal_duration * 25  # Scale duration score based on the actual duration
    
    # Age factor (younger age may contribute to higher score)
    if 20 <= age <= 30:
        age_score = 10  # High score for young age
    elif 30 < age <= 40:
        age_score = 7
    else:
        age_score = 5  # Lower score for older age

    # Summing up the scores
    total_score = bmi_score + heart_rate_score + temp_score + duration_score + age_score

    # Normalize the score to be out of 100
    fitness_score = min(total_score, 100)
    return fitness_score

fitness_score = calculate_fitness_score(weight, height, age, gender, heart_rate, body_temp, duration, bmi)

# Display Fitness Score
st.metric("🏆 Your Fitness Score", f"{fitness_score:.2f}")

# Create a Gauge Chart to visualize the Fitness Score
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=fitness_score,
    title={'text': "Fitness Score"},
    gauge={
        'axis': {'range': [None, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 50], 'color': "red"},
            {'range': [50, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "green"}
        ],
    }
))
st.plotly_chart(fig)

# Fitness Insights based on Score
st.subheader("💡 Fitness Insights")

if fitness_score > 75:
    st.success("✅ Great job! You have an excellent fitness level!")
elif fitness_score > 50:
    st.warning("⚠️ Good! But there's room for improvement.")
else:
    st.error("🚨 It looks like you may need to focus more on fitness and exercise.")

# Correlation Between Sleep, Stress, and BMI
st.markdown("<p class='big-font'>📈 Sleep, Stress & BMI Correlation</p>", unsafe_allow_html=True)
stress_bmi_fig = px.scatter(sleep_df, x='Stress Level', y='BMI Category', color='Sleep Duration', 
                            title='💡 Impact of Stress & Sleep on BMI')
st.plotly_chart(stress_bmi_fig)

# Sleep Analysis Insights
st.subheader("🌙 Sleep & Stress Insights")
if sleep_duration < 6:
    st.warning("⚠️ You may not be getting enough sleep. Consider improving your sleep routine.")
elif 6 <= sleep_duration <= 8:
    st.success("✅ You have a healthy sleep duration! Keep it up.")
else:
    st.warning("⚠️ Too much sleep might affect productivity and health.")

if stress_level > 7:
    st.error("🚨 High stress levels detected! Try meditation, exercise, or relaxation techniques.")
    

# Create a table for user input parameters
user_params = {
    "Age": [age],
    "Weight (kg)": [weight],
    "Height (cm)": [height],
    "Heart Rate (bpm)": [heart_rate],
    "Body Temp (°C)": [body_temp],
    "Exercise Duration (mins)": [duration],
    "Gender": [gender]
}

# Convert to DataFrame
user_df = pd.DataFrame(user_params)

# Display the table
st.subheader("Your Input Parameters:")
st.table(user_df)


# Show Sample Data
st.subheader("📂 Other Results ")
st.write(fitness_df.head())


# Diet Recommendation Section
st.subheader("🥗 Personalized Diet Recommendations")
if bmi < 18.5:
    diet_type = "paleo"
elif 18.5 <= bmi < 25:
    diet_type = "mediterranean"
elif 25 <= bmi < 30:
    diet_type = "dash"
else:
    diet_type = "keto"

# Display the chosen diet type based on BMI
st.write(f"Filtering diet type: {diet_type}")

# Filter diet recommendations
diet_recommendations = diet_df[diet_df["Diet_type"].str.lower().str.contains(diet_type.lower(), na=False)]

# Visualizing the diet type distribution
diet_counts = diet_df["Diet_type"].value_counts().reset_index()
diet_counts.columns = ["Diet Type", "Count"]

# Plot the pie chart
st.subheader("📊 Diet Type Distribution")
fig = px.pie(diet_counts, names="Diet Type", values="Count", title="Diet Type Distribution")
st.plotly_chart(fig)

# Display a few diet recommendations
st.write(f"Based on your BMI ({bmi}), we recommend a {diet_type} diet.")
st.subheader("🍽️ Diet Recommendations")
diet_recommendations_display = diet_df[diet_df["Diet_type"].str.lower().str.contains(diet_type.lower(), na=False)].head(5)

st.table(diet_recommendations_display[["Recipe_name", "Cuisine_type", "Protein(g)", "Carbs(g)", "Fat(g)"]])

# Generate Personalized Tips
tips = []

# BMI-Based Tips
if bmi < 18.5:
    tips.append("Increase your calorie intake with nutritious meals.")
    tips.append("Incorporate strength training exercises to build muscle.")
    tips.append("Snack on protein-rich foods like nuts and yogurt.")
    tips.append("Drink plenty of water to stay hydrated.")
    tips.append("Ensure you're getting enough sleep to aid muscle recovery.")
elif 18.5 <= bmi < 25:
    tips.append("Maintain a balanced diet with proteins, carbs, and healthy fats.")
    tips.append("Engage in regular physical activity like walking or jogging.")
    tips.append("Stay hydrated and avoid sugary drinks.")
    tips.append("Incorporate flexibility exercises like yoga.")
    tips.append("Monitor portion sizes to maintain a healthy weight.")
elif 25 <= bmi < 30:
    tips.append("Reduce refined carbs and sugars in your diet.")
    tips.append("Increase daily physical activity to at least 30 minutes.")
    tips.append("Try meal prepping to avoid unhealthy food choices.")
    tips.append("Get enough fiber from vegetables and whole grains.")
    tips.append("Consider mindful eating to avoid overeating.")
else:
    tips.append("Follow a structured diet plan with lower carbs.")
    tips.append("Engage in high-intensity workouts like interval training.")
    tips.append("Track your calorie intake to manage weight effectively.")
    tips.append("Prioritize sleep to regulate metabolism.")
    tips.append("Stay consistent with your health and fitness routine.")

# Sleep-Based Tips
if sleep_duration < 6:
    tips.append("Try to get at least 7-8 hours of sleep for better recovery.")
    tips.append("Avoid screens before bedtime to improve sleep quality.")
    tips.append("Maintain a consistent sleep schedule daily.")
    tips.append("Limit caffeine intake in the evening.")
    tips.append("Create a relaxing bedtime routine.")

# Activity Level-Based Tips
if activity_level in ["Sedentary", "Light"]:
    tips.append("Incorporate at least 30 minutes of daily physical activity.")
    tips.append("Take short walks after meals to boost digestion.")
    tips.append("Use stairs instead of elevators whenever possible.")
    tips.append("Stretch regularly to improve flexibility.")
    tips.append("Consider desk exercises if you work long hours sitting.")

# Display the tips
st.subheader("💡 Useful Tips for You")
for tip in tips[:5]:  # Show only the top 5 tips
    st.write(f"✅ {tip}")
