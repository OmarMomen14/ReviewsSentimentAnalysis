import flask
import pickle
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import youtokentome as yttm
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import to_categorical
import re, unicodedata
import pickle
import time




app = flask.Flask(__name__, template_folder='templates')

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
# this is the dictionary created by youtokentome library consisting of the most repeated 5000 words from the cleaned data
bpe = yttm.BPE(model="model7")
# the different types of personalities

stemmer = PorterStemmer()         # Initialize both stemmer and lemmatiser
lemmatiser = WordNetLemmatizer()
numbers= ['1','2','3','4','5','6','7','8','9' ]

model = tf.keras.models.load_model('my_modelX.h5')


# Remove URL from data string
def remove_URL(data):
    return re.sub(r"http\S+", "", data)


# Remove non-ASCII characters from list of tokenized words
def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


# Convert to lowercase from list of tokenized words
def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


# Remove punctuations from list of toknized words
def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        new_word = re.sub(r'\d+', '', new_word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


# Remove stopwords from list of toknized words
def remove_stopwords(words):
    new_words = []
    for word in words:
# added not in types to remove the class names from the people posts so it wont make the model use it to make predicitions
        if word not in stopwords.words('english') and word not in numbers:
            new_words.append(word)
    return new_words


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words


@app.route('/',methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
	if flask.request.method == 'POST':
		review = flask.request.form['review']
		cleaned = remove_URL(review)
		cleaned = nltk.word_tokenize(cleaned)
		cleaned = normalize(cleaned)
		for x in range(len(cleaned)):
			cleaned[x] = lemmatiser.lemmatize(cleaned[x])
		sentence = ' '.join(cleaned)
		encoded_sentence = bpe.encode(sentence, output_type=yttm.OutputType.ID)
		padded_sentence = sequence.pad_sequences([encoded_sentence], maxlen=30)
		prediction = model.predict_classes(padded_sentence)[0]
		print("class is: ",prediction)
		print(review)
		result = " "
		if prediction == 0:
			result = "Negative Review"
			f = open("negative.txt","a")
			review = review + "\n"
			f.write(review) 
			f.close() 
		elif prediction == 1:
			result = "Positive Review"
			f = open("positive.txt","a")
			review = review + "\n"
			f.write(review) 
			f.close()
		return flask.render_template('main.html',inputText=review, result=result)



@app.route('/positive')
def positive():
	if flask.request.method == 'GET':
		positiveFile = open("positive.txt", "r")
		return(flask.render_template('positive.html', lines = positiveFile))

@app.route('/negative')
def negative():
	if flask.request.method == 'GET':
		negativeFile = open("negative.txt", "r")
		return(flask.render_template('negative.html', lines= negativeFile))

if __name__ == '__main__':
    app.debug = True
    app.run()