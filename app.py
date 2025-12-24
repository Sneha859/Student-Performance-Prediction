import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("student_data.csv")

X = data[["study_hours", "attendance", "previous_score"]]
y = data["pass"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Performance Prediction")

st.write("Enter student details to predict pass or fail")

study_hours = st.number_input("Study Hours per Day", 0, 12)
attendance = st.number_input("Attendance (%)", 0, 100)
previous_score = st.number_input("Previous Exam Score", 0, 100)

if st.button("Predict"):
    new_student = pd.DataFrame(
        [[study_hours, attendance, previous_score]],
        columns=["study_hours", "attendance", "previous_score"]
    )
    
    prediction = model.predict(new_student)
    
    if prediction[0] == 1:
        st.success("✅ Prediction: PASS")
    else:
        st.error("❌ Prediction: FAIL")
