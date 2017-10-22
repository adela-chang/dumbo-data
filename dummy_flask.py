from flask import Flask
from flask import request
import subprocess
from subprocess import PIPE
import nltk
from flask_uwsgi_websocket import GeventWebSocket
from werkzeug.datastructures import ImmutableMultiDict
from nltk.sentiment.util import demo_vader_instance
from nltk.sentiment.util import extract_unigram_feats
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS
from flask_cors import cross_origin
from string import digits
from flask import jsonify

import librosa
app = Flask(__name__)
CORS(app)


def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

@app.route('/')
def hello_world():
    return 'Hello World!'


#ws = GeventWebSocket(app)

#@ws.route('/websocket')
def audio(ws):
   first_message = True
   total_msg = ""
   sample_rate = 0

   while True:
      msg = ws.receive()

      if first_message and msg is not None: # the first message should be the sample rate
         sample_rate = getSampleRate(msg)
         first_message = False
         continue
      elif msg is not None:
         audio_as_int_array = numpy.frombuffer(msg, 'i2')
         doSomething(audio_as_int_array)
      else:
         break
   f = open('working.m4a', 'w')
   f.write(msg)
   text_output = recognition('working.m4a')
   scores, male, female = sexism_sentiment(text_output)
   return scores, male, female

def recognition(file_input):
    process = subprocess.check_output('python speech-to-text-wavenet/recognize.py --file %s' %(file_input), shell=True)
    return process

def sexism_sentiment(input_text):
    #got_a_feeling = demo_vader_instance(input_text)
    darth = SentimentIntensityAnalyzer()
    got_a_feeling = darth.polarity_scores(input_text)
    male_unigrams = ['he','his','him']
    female_unigrams = ['she','hers','her']
    male_feats = extract_unigram_feats(input_text,male_unigrams)
    print(male_feats)
    male_in = any('True' in x for x in male_feats)
    print(male_in)
    female_feats = extract_unigram_feats(input_text,female_unigrams)
    print(female_feats)
    female_in = any('True' in x for x in female_feats)
    print(female_in)
    #female_in = any(extract_unigram_feats(input_text,female_unigrams))
    return got_a_feeling, female_in, male_in

#@cross_origin()
@app.route('/send', methods=['GET', 'POST'])
@cross_origin()
def parse_request():
    data = request.data
    print(data)
    #data = request.stream
    #print(data)
    #data = dict(request.form)
    #print(data)
    f = open('working.flac', 'w')
    f.write(data)
    #librosa.output.write_wav('working.wav', data, sr=14000)
    text_output = recognition('working.flac')
    result = text_output.translate(None, digits)
    print(result)
    scores, female, male = sexism_sentiment(result)
    output = jsonify(sentiment_compound=scores['compound'],sentiment_pos=scores['pos'],sentiment_neg=scores['neg'],sentiment_neu=scores['neu'],female_in=female,male_in=male)
    return output

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=9000, debug=False)
