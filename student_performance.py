import pandas as pd

# Load dataset
data = pd.read_csv("student_data.csv")

# Display data
print(data)


from sklearn.model_selection import train_test_split

# Features and target
X = data[["study_hours", "attendance", "previous_score"]]
y = data["pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully")


from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


import pandas as pd

new_student = pd.DataFrame(
    [[6, 82, 68]],
    columns=["study_hours", "attendance", "previous_score"]
)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction for new student: PASS")
else:
    print("Prediction for new student: FAIL")


import joblib

joblib.dump(model, "student_model.pkl")
print("Model saved successfully")
