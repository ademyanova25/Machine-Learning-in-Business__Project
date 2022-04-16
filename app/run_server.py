import dill
import pandas as pd
from flask import render_template, redirect, url_for, Flask, request, jsonify
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation

english_stopwords = stopwords.words('english')
stemmer = SnowballStemmer(language='english')


# initialize our Flask application and the model
app = Flask(__name__, template_folder="templates")

model_path = "/app/app/model/PassiveAggressiveClassifier_pipeline.dill"
with open(model_path, 'rb') as in_strm:
    model = dill.load(in_strm)


def transformer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    filtered_tokens = []
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    for token in tokens:
        if token not in english_stopwords and token != " " and token.strip() not in punctuation:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in english_stopwords]
    new_text = ' '.join(stems)

    return new_text


def send_json(x):
    text = x

    body = {
        'text': text
        }
    my_url = 'http://127.0.0.1:5000/' + '/predict'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(my_url, json=body, headers=headers)
    return response.json()['predictions']


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/form_predict', methods=['GET', 'POST'])
def form_predict():
    if request.method == 'POST':
        text = str(request.form.get('text'))
        text = transformer(text)
        response = send_json(text)
        return redirect(url_for('predicted', response=response))
    return render_template("form.html")


@app.route('/predicted/<response>')
def predicted(response):
    dict_resp = {
        '0': 'Fake',
        '1': 'Real'
    }
    print('This article is', dict_resp[response])
    return render_template('predicted.html', response=dict_resp[response])


@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    text = ""
    request_json = request.get_json()

    if request_json["text"]:
        text = request_json['text']

    predicts = model.predict(pd.DataFrame({"text": [text]})).tolist()
    data["predictions"] = predicts[0]
    data["text"] = text

    # indicate that the request was a success
    data["success"] = True
    print('OK')

    # return the data dictionary as a JSON response
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
