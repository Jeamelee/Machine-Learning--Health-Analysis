import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with your actual data file path)
data = pd.read_csv("Prostate_Cancer.csv")

# Assume 'Cancer' column contains labels (0 for no cancer, 1 for cancer)
Diagnosis = {'B': 0, 'M': 1}
data['diagnosis_result'] = data['diagnosis_result'].map(Diagnosis)

# Features (X) and target (y)
X = data.drop(columns=['diagnosis_result', 'id'])
y = data['diagnosis_result']

# Filter out rows with NaN values in both X and y
finite_filter = np.isfinite(X).all(axis=1) & np.isfinite(y)
X_finite = X[finite_filter]
y_finite = y[finite_filter]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_finite, y_finite, test_size=0.4, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.show()

# Print the R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")


model = DecisionTreeClassifier(max_depth=4, random_state=0)
model.fit(X_train, y_train)

export_graphviz(model, out_file="prostate.dot", class_names=["benign", "malignant"],
                feature_names=X.columns, impurity=False, filled=True)


# Ask questions
print("Answer the following questions (yes/no):")
age = input("Are you above 50 years old? ")
psa_level = input("Have you had elevated PSA levels? ")
family_history = input("Do you have a family history of prostate cancer? ")
urinary_symptoms = input("Do you experience frequent urination or difficulty urinating? ")
pain_discomfort = input("Do you have pain or discomfort in the pelvic area? ")
weight_loss = input("Have you experienced unexplained weight loss recently? ")
blood_in_urine = input("Have you noticed blood in your urine? ")
difficulty_maintaining_erection = input("Do you have difficulty maintaining an erection? ")

# Convert answers to binary (yes=1, no=0)
answers = [1 if ans.lower() in ["yes", "y", "Yes", "sure"] else 0 for ans in [age, psa_level, family_history, urinary_symptoms,
                                                        pain_discomfort, weight_loss, blood_in_urine,
                                                        difficulty_maintaining_erection]]

# Predict using the model
diagnosis_prediction = model.predict([answers])[0]

# Use the model to predict the probability of prostate cancer
proba = model.predict_proba([answers])

# Print the probability of prostate cancer
print(f"The probability of prostate cancer is {proba[0][1] * 100:.2f}%.")

if diagnosis_prediction == 1:
    print("\nBased on your answers, there's a possibility of prostate cancer. Please consult a doctor.")
else:
    print("\nYour answers suggest a lower risk of prostate cancer, but regular screenings are still important.")
