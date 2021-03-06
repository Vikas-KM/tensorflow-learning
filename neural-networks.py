import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my Dog',
    'i love my Cat',
    'you love my Dog',
    'Do you love my dog?'
]

# tokenizer = Tokenizer(num_words=100)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_data = [
    'i really love my dog',
    'my dog loves my cat'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
