from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Train dummy model
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0])

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save using scikit-learn 1.6.1
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)