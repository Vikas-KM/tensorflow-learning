import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import sys

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

print(sys.version)
print(tf.__version__)
