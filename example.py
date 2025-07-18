
# Predict whether a person will buy a product based on their age and salary
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the data
data = {
  'Age': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
  'Salary': [ 20000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000],
  'purchased': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] # Target: 1 = Yes, 0 = No
}
df = pd.DataFrame(data)

# Step 2: Split features (X) and target (y)
X = df[['Age', 'Salary']] # Features
y = df['purchased']       # Target

# Step 3: Train-Test Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Step 4: Build and Train the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("y_test:", y_test)
print("y_pred:", y_pred)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['Age', 'Salary'], class_names=['No', 'Yes'], filled=True)
plt.show()

# Step 8: Predict new value
new_person = [[33, 58000]]  # Age: 33, Salary: 58000
prediction = model.predict(new_person)
print("\nWill the person buy the product?", "Yes ✅" if prediction[0] == 1 else "No ❌")
