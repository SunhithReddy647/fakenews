from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = model.predict([message])
        return render_template('predict.html', prediction=pred)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)