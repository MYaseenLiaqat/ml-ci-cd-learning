import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
heart_disease = pd.read_csv("Heart Disease UCI.csv")

# Features and target
X = heart_disease.drop("condition", axis=1)
y = heart_disease["condition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
clf = RandomForestClassifier()

# Train model
clf.fit(X_train, y_train)

# Evaluate model
score = clf.score(X_test, y_test)

print("Model Accuracy:", score)

# CI validation check
assert score > 0.70