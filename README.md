# 🤖 Logistic Regression: Predicting Purchase Decisions from Age & Salary

This project demonstrates how to build a **Logistic Regression classification model** using `scikit-learn` to predict whether a person will purchase a product based on their **age** and **salary**. Perfect for beginners to learn how classification works with real-world-like data!

---

## 📌 Project Overview

- **Machine Learning Task**: Binary Classification
- **Model Used**: Logistic Regression
- **Libraries**: Python, pandas, matplotlib, scikit-learn
- **Input Features**: Age, Salary
- **Target Variable**: Purchased (1 = Yes, 0 = No)

---

## 📊 Dataset

```python
data = {
  'Age': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
  'Salary': [20000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000],
  'purchased': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
}
Age	Salary	Purchased
21	20000	0
22	22000	0
23	23000	1
24	24000	1
25	25000	0
26	26000	1
27	27000	0
28	28000	1
29	29000	0
30	30000	1

🔧 Steps Covered
✅ Step 1: Load the Data
import pandas as pd

data = {
  'Age': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
  'Salary': [20000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000],
  'purchased': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

✅ Step 2: Preprocess and Split the Data
from sklearn.model_selection import train_test_split

X = df[['Age', 'Salary']]
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

✅ Step 3: Train Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

✅ Step 4: Make Predictions and Evaluate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred)

✅ Step 5: Visualize the Data (Optional)
import matplotlib.pyplot as plt

plt.scatter(df['Age'], df['Salary'], c=df['purchased'], cmap='bwr')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary - Purchased')
plt.grid(True)
plt.show()

📈 Output Example
Accuracy: 1.0
Confusion Matrix:
[[2 0]
 [0 0]]
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         2
           1       0.00      0.00      0.00         0
Note: With a small dataset, results may vary. This is for educational purpose.

✅ What You’ll Learn
How to perform binary classification with Logistic Regression

How to split data and evaluate model performance

How to use accuracy, confusion matrix, and classification report

How to visualize classification data in 2D

🚀 Future Improvements
Use a larger and more balanced dataset

Add feature scaling (e.g., StandardScaler)

Try other models: Decision Tree, KNN, SVM, etc.

Plot decision boundaries (for visual learners)

🛠 Tech Stack
Python 3

pandas

matplotlib

scikit-learn

🙌 Let’s Connect
If you liked this project, feel free to connect, fork, or star ⭐ the repo!

