from flask import Flask, render_template, request, redirect, url_for
from main import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text')
        prediction = predict_sentiment(text)
        return render_template('results.html', prediction=prediction)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
