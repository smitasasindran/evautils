import tensorflow as tf

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
IMAGE_DEPTH = 3
NUM_CLASSES = 10


# ENCODING
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecord(data, labels, tfrecords_filename):
  """Converts a file to TFRecords."""
  print('Generating %s' % tfrecords_filename)
  with tf.python_io.TFRecordWriter(tfrecords_filename) as record_writer:
    num_entries_in_batch = len(labels)
    for i in range(num_entries_in_batch):
      example = tf.train.Example(features=tf.train.Features(
        feature={
          'image': _bytes_feature(data[i].tobytes()),
          'label': _int64_feature(labels[i])
        }))
      record_writer.write(example.SerializeToString())


""" Usage
train_tfrecords_filename = 'TrainCifar10.tfrecords'
test_tfrecords_filename = 'TestCifar10.tfrecords'
_convert_to_tfrecord(x_train, y_train,train_tfrecords_filename)
_convert_to_tfrecord(x_test,y_test,test_tfrecords_filename)
"""

#DECODING
def parse_record(serialized_example, isTraining = True):
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })
  
  image = features['image']
  image = tf.decode_raw(image, tf.float32)

  if(isTraining):
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
  else:
    image = tf.reshape(image, [32, 32, IMAGE_DEPTH])
  
  label = tf.cast(features['label'], tf.int64)
  return image, label

def generate_input_fn(file_name, isTraining = True):
  dataset = tf.data.TFRecordDataset(filenames=file_name)
  dataset = dataset.map(lambda x: parse_record(x, isTraining))
  return dataset  



