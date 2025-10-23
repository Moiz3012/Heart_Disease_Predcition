from flask import Flask, render_template, request
import numpy as np
import pickle  # if you want to load saved model, optional

app = Flask(__name__)

# Example: replace these with your trained model parameters
w_final = np.array([-0.61625282, 0.04969199, -0.22803034, 0.15642186, -0.08263041, -0.33069563,
                    0.67456641, -1.19146982, 1.46848136, 1.99883652, -0.12777168, 0.36130379,
                    0.04328583])
b_final = 4.11

# Means and stds from training (for normalization)
X_mean = np.array([54.43, 0.68, 1.56, 131.69, 246.26, 0.15, 0.53, 149.6, 0.33, 1.04, 1.4, 0.67, 2.31])
X_std = np.array([9.04, 0.47, 0.97, 17.54, 51.83, 0.36, 0.52, 22.9, 0.47, 1.16, 0.61, 1.02, 0.61])

def sigmoid(x, w, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        try:
            # Read inputs and convert to float
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])

            x = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal])

            # Normalize
            x_norm = (x - X_mean) / X_std

            # Predict
            pred = sigmoid(x_norm, w_final, b_final)
            decision = "Heart Disease" if pred >= 0.5 else "No Heart Disease"

            result = f"💓 Probability: {pred*100:.2f}% → {decision}"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
