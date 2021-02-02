import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import csv
from bs4 import BeautifulSoup
import string

sentences = []
labels = []

stopwords = ['a', ..., 'yourselves']
table = str.maketrans('', '', string.punctuation)

with open('./data/text_emotion.csv', 'r', encoding='UTF-8') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        labels.append(row[1])
        sentence = row[3].lower()
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace('-', ' - ')
        sentence = sentence.replace('/', ' / ')
        soup = BeautifulSoup(sentence, 'html.parser')
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ''
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + ' '
        sentences.append(filtered_sentence)

print(len(sentences))
print(sentences[2])
print(len(labels))
print(labels[2])

# total 40,000 so lets split 30K for training and 10K for testing
training_size = 30000
training_sentences = sentences[1:training_size]
testing_sentences = sentences[1:training_size]
training_labels = sentences[training_size:]
testing_labels = sentences[training_size:]

vocab_size = 30000
max_length = 15
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type,
)

print(training_sequences[0])
print(training_padded[0])

print(word_index)
