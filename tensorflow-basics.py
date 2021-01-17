import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('tensorflow version ', tf.__version__)

# INITIALIZATION OF TENSORS
x = tf.constant(4)
print(x)
x = tf.constant(4, shape=(1, 1))
print(x)
x = tf.constant(4, shape=(2,), dtype=tf.float32)
print(x)
x = tf.constant(44, shape=(2, 2))
print(x)

x = tf.ones((3, 3))
print(x)

x = tf.zeros((2, 3))
print(x)

x = tf.eye(3)
print(x)

# Identity matrix
x = tf.eye(2, 3)
print(x)

# this is standard normal distribution / Gaussian
x = tf.random.normal((2, 3), mean=0, stddev=1)
print(x)

# for uniform distribution
x = tf.random.uniform((2, 3), minval=0, maxval=1)
print(x)

# upper value is exclusive
x = tf.range(9)
print(x)

# limit is endpoint, delta = steps size
x = tf.range(start=1, limit=10, delta=2)
print(x)

# casting from one type to another
x = tf.cast(x, dtype=tf.float32)
print(x)

# MATHEMATICAL OPERATIONS
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

# Adding
z = tf.add(x, y)
print(z)
z = x + y
print(z)

# Subtract
z = tf.subtract(x, y)
print(z)
z = x - y
print(z)

# Multiply
z = tf.multiply(x, y)
print(z)
z = x * y
print(z)

# Divide
z = tf.divide(x, y)
print(z)
z = x / y
print(z)

# dot product
print(x)
print(y)
z = tf.tensordot(x, y, axes=0)
print(z)

# both produce same results
z = tf.tensordot(x, y, axes=1)
print(z)
z = tf.reduce_sum(x * y, axis=0)
print(z)

z = x ** 5
print(z)

# matrix multiplication
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y)
print(z)
z = x @ y
print(z)

# INDEXING
x = tf.constant([0, 1, 1, 2, 4, 5, 7, 3, 1])
# prints all elements
print(x[:])
# print from 1 to end
print(x[1:])
# print [1,1]
print(x[1:3])
# print alternate values
print(x[::2])
# reverse order
print(x[::-1])

# print 0 and 2 values
indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
# print first row
print(x[0])
# other way
print(x[0, :])
# first two rows
print(x[0:2, :])

# RESHAPING
x = tf.range(9)
print(x)

x = tf.reshape(x, shape=(3, 3))
print(x)

x = tf.transpose(x)
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)
