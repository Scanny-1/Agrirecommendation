from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

app = Flask(__name__)

print("Loading dataset and training models...\n")

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Encode crop labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label_encoded']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

accuracies = {}

print("==================== Model Evaluation (Terminal Output) ====================\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    accuracies[name] = round(acc * 100, 2)

    print(f"\n📌 Results for {name}")
    print("-------------------------------------------")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Accuracy Comparison Graph
plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Comparison of Model Accuracy")
plt.show()

# Best model for predictions
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def recommend():
    crop_result = None

    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        area = float(request.form['area'])   # NEW FIELD

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = best_model.predict(data)[0]
        crop_name = le.inverse_transform([prediction])[0]

        # Dummy values (Replace later with actual logic)
        expected_yield = round(area * 250, 2)   # per acre yield assumption
        profit = expected_yield * 25             # Rs 25/kg
        urea = round(area * 50, 2)
        dap = round(area * 30, 2)
        mop = round(area * 20, 2)

        crop_result = {
            "crop": crop_name,
            "area": area,
            "yield": expected_yield,
            "profit": profit,
            "urea": urea,
            "dap": dap,
            "mop": mop
        }

    return render_template('index.html', result=crop_result, accuracies=accuracies)

if __name__ == '__main__':
    app.run(debug=True)
