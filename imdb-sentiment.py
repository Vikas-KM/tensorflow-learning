import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from bs4 import BeautifulSoup
import string

imdb_sentences = []
imdb_train = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
# for item in imdb_train:
#     imdb_sentences.append(str(item['text']))

# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(imdb_sentences)
# sequences = tokenizer.texts_to_sequences(imdb_sentences)
# word_index = tokenizer.word_index

# print(word_index)

# most of the words in the index are stopwords and html tags

stopwords = ['a', ..., 'yourselves']
print(stopwords)
table = str.maketrans('', '', string.punctuation)
print(table)

for item in imdb_train:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    sentence = sentence.replace(',', ' , ')
    sentence = sentence.replace('.', ' . ')
    sentence = sentence.replace('-', ' - ')
    sentence = sentence.replace('/', ' / ')
    words = sentence.split()
    filtered_sentence = ''
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + ' '
    imdb_sentences.append(filtered_sentence)

tokenizer = Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
word_index = tokenizer.word_index
print(word_index)
