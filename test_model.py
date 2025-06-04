# test_model.py
import pickle
model = pickle.load(open("model.pkl", "rb"))
print(model.predict([[0, 0]]))  # Example prediction with dummy data
input("Press Enter to exit...")  # Keep the console open to see the output
