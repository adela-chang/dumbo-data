from flask import Flask
from flask import request
import subprocess
from subprocess import PIPE
import nltk
#import librosa
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

def recognition(file_input):
    process = subprocess.call('python speech-to-text-wavenet/recognize.py --file %s' %(file_input), shell=True)
    return process

def sexism_sentiment(input_text):
    got_a_feeling = nltk.sentiment.util.demo_vader_instance(input_text)
    male_unigrams = ['he','his','him']
    female_unigrams = ['she','hers','her']
    male_in = any(nltk.sentiment.util.extract_unigram_feats(input_text,male_unigrams))
    female_in = any(nltk.sentiment.util.extract_unigram_feats(input_text,female_unigrams))
    return got_a_feeling, male_in, female_in

@app.route('/send', methods=['GET', 'POST'])
def parse_request():
    data = request.stream
    f = open('working.m4a', 'w')
    f.write(data)
    text_output = recognition('working.m4a')
    scores, male, female = sexism_sentiment(text_output)
    return scores, male, female

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=9000, debug=False)
