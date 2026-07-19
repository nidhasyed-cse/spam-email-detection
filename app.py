from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

model = None
cv = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():

    global model, cv

    if 'dataset' not in request.files:
        return render_template(
            'index.html',
            msg="No file selected"
        )

    file = request.files['dataset']

    if file.filename == '':
        return render_template(
            'index.html',
            msg="Please choose a CSV file"
        )

    print("Filename:", file.filename)

    try:
        data = pd.read_csv(file)
        print(data.head())
    except Exception as e:
        print("ERROR:", e)
        return render_template(
            'index.html',
            msg=f"Error: {e}"
        )

    x = data["message"]
    y = data["label"]

    cv = CountVectorizer()
    x_vector = cv.fit_transform(x)

    model = MultinomialNB()
    model.fit(x_vector, y)

    return render_template(
        'index.html',
        msg="Dataset Loaded Successfully"
    )

@app.route('/predict', methods=['POST'])
def predict():

    global model, cv

    if model is None:
        return render_template(
            'index.html',
            msg="Please Load Dataset First"
        )

    email = request.form['email']

    data_vector = cv.transform([email])

    prediction = model.predict(data_vector)

    return render_template(
        'index.html',
        prediction=prediction[0]
    )

if __name__ == '__main__':
    app.run(debug=True)