import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import export_text
import joblib  # Use joblib for model saving and loading

# Step 1: Data Preprocessing
data = pd.read_csv("dataset.csv")

# Select relevant features for classification
features = ['Cpu Utilization', 'Network Latency (milliseconds)','Average IOPS']
X = data[features]
y = data['classification']

# Step 4: Model Selection and Training (Decision Tree Classifier)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the trained model to a file using joblib
joblib.dump(classifier, 'model/decision_tree_model.joblib')

# Step 8.1: Optionally, you can save the trained features for later use
joblib.dump(features, 'model/model_features.joblib')

# Step 8.2: Visualization of Decision Tree (Optional)
tree_rules = export_text(classifier, feature_names=features)
print("Decision Tree Rules:\n", tree_rules)
