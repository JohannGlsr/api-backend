from flask import Flask, request, jsonify
import nltk
import string
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pickle

# Initialisation des stopwords /  words / lemmatization
stopwords = nltk.corpus.stopwords.words('english') # Mots à supprimer
words = set(nltk.corpus.words.words()) # Totalités des mots de la langue
lemmatizer = WordNetLemmatizer() # Pour préserver la racine du mots

def Preprocess_Sentence(Sentence):        
    # Enlever la ponctuation
    Sentence = "".join([i.lower() for i in Sentence if i not in string.punctuation])
    # Enlever les chiffres
    Sentence = ''.join(i for i in Sentence if not i.isdigit())
    # Tokenization : Transformer les phrases en liste de tokens (en liste de mots)
    Sentence = nltk.tokenize.word_tokenize(Sentence)
    # Enlever les stopwords
    Sentence = [i for i in Sentence if i not in stopwords]
    # Enlever les majuscules
    Sentence = ' '.join(w for w in Sentence if w.lower() in words or not w.isalpha())

    return Sentence 

# Chargement du tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Chargement du modèle
loaded_model = load_model('advance_model.h5')

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    phrase = request.form['phrase']
    sequence = Preprocess_Sentence(phrase)
    sequence = tokenizer.texts_to_sequences([sequence])
    while len(sequence[0]) < 35:
        sequence[0].insert(0, 0)
    prediction = loaded_model.predict(sequence)
    stringprediction = str(prediction[0][0])
    return jsonify({'prediction': stringprediction})

if __name__ == '__main__':
    app.run()