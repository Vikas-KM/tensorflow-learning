import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# (train_images, train_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('fashion_mnist', split=['train','test'], batch_size=-1, as_supervised=True))
# (training_images, training_labels), (test_images, test_labels) =  tfds.as_numpy(tfds.load('fashion_mnist',split = ['train', 'test'], batch_size=-1, as_supervised=True))


image, label = tfds.as_numpy(tfds.load(
    'fashion_mnist',
    split='test',
    batch_size=-1,
    as_supervised=True,
))

print(type(image), image.shape)


# import tensorflow_datasets as tfds
#
# # Construct a tf.data.Dataset
# ds = tfds.load('mnist', split='train', shuffle_files=True)
#
# # Build your input pipeline
# ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
# for example in ds.take(1):
#   image, label = example["image"], example["label"]

def augumentation(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='nearest')
    return image, label


data = tfds.load('fashion_mnist', split='test', as_supervised=True)

test = data.map(augumentation)
