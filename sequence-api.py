import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my Dog',
    'my cat loves me',
    'Does my dog love me?',
    'my cat loves my mother more'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print('word index')
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print('sequences')
print(sequences)

padded = pad_sequences(sequences)
print('Padded')
print(padded)

test_data = [
    'i really love my dog',
    'my cat likes my father'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print('test sequences')
print(test_seq)

test_pad = pad_sequences(test_seq)
print('test padded')
print(test_pad)
