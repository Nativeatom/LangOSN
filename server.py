from flask import Flask
from flask import g, request, jsonify
from multiprocessing import Process 
import requests, json, random

# app = Flask(__name__, static_url_path='')
app = Flask(__name__)
app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False

@app.teardown_request
def teardown_request(error):
    # record error
    print('teardown_request：%s' % error)

@app.route('/')
def index():
    return app.send_static_file('HelloWorld.html'), 400

@app.route('/loadLangModel')
def loadLangModel():
    lang = request.args.get('lang')
    print('load {} Model'.format(lang))
    # TODO: load model of the language specified
    return 'load {} Model'.format(lang)

@app.route('/sendNewToken', methods = ['GET', 'POST'])
def sendNewToken():
    if request.method == 'POST':
        lang = request.form['lang']
        token = request.form['Token']
        print("newToken {} in {}".format(token, lang))
        result = random.choice(['correct', 'misspelled'])
        result2front = {'lang': lang, 'Token': token, 'result': result}
    return jsonify(result2front)

@app.route('/sendTokenCorrection', methods = ['GET', 'POST'])
def sendTokenCorrection():
    if request.method == 'POST':
        lang = request.form['lang']
        token = request.form['Token']
        decision = request.form['userDecision']
        print("user thinks {} is {} in {}".format(token, decision, lang))
        result = 'received'
        result2front = {'lang': lang, 'Token': token, 'result': result}
    return jsonify(result2front)

@app.route('/receiver', methods = ['GET', 'POST'])
def worker():
    if request.method == "POST":
        # read json + reply
        # data = request.get_json(force=True)
        print("header: ", request.header)
        print("form: ", request.form)
        data = request.get_data()
        result = ''

        # loop over every row
        result += data['Token'] + '(' + data['lang'] + ')' + '\n'

        return result



if __name__ == '__main__':
    # threaded=True?
    app.run(host="localhost", port=8900, debug=True)

