import numpy as np 
import tensorflow as tf 
import pdb


# tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    state_size = 3
    action_size = 2
    output_size = state_size + action_size**2 + state_size*action_size + 2*state_size**2

    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=output_size)

    matrixTransformA = tf.matrix_band_part(tf.reshape(dense[-1,:state_size*state_size],[-1,state_size,state_size]),0,-1)
    idx = state_size*state_size
    matrixDiagonalA = tf.reshape(tf.diag(dense[-1,idx:idx+state_size]),[-1,state_size,state_size])
    idx += state_size
    matrixF = tf.reshape(dense[-1,idx:idx+state_size*action_size],[-1,state_size,action_size])
    idx += state_size*action_size
    matrixGramQ = tf.reshape(dense[-1,idx:idx+state_size*state_size],[-1,state_size,state_size])
    idx += state_size*state_size
    matrixGramR = tf.reshape(dense[-1,idx:],[-1,action_size,action_size])

    predictions = {
      "A": tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA),name="matrixA"),
      "B": tf.matmul(matrixTransformA,matrixF,name="matrixB"),
      "Q": tf.matmul(matrixGramQ,matrixGramQ,transpose_a=True,name="matrixQ"),
      "R": tf.matmul(matrixGramR,matrixGramR,transpose_a=True,name="matrixR"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00)
        train_op = optimizer.minimize(
            loss=dense[0][0],
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=dense[0][0], train_op=train_op)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    tensors_to_log = {
        "A": "matrixA",
        "B": "matrixB",
        "Q": "matrixQ",
        "R": "matrixR"}

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data[0:3]},
        y=train_labels[0:3],
        batch_size=3,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1)


    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data[0:10]},
        num_epochs=1,
        shuffle=False)

    predictions = list(mnist_classifier.predict(input_fn=predict_input_fn))
    pdb.set_trace()

if __name__ == "__main__":
    tf.app.run()