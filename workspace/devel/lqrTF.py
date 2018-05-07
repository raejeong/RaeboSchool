import numpy as np 
import tensorflow as tf 
import pdb


tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array

    small_number = 0.001
    state_size = 3
    action_size = 2
    output_size = state_size + action_size**2 + state_size*action_size + 2*state_size**2

    x = tf.placeholder(tf.float32, [None, 784])

    input_layer = tf.reshape(x,[-1,28,28,1])

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
    dense = tf.where(tf.abs(dense)<small_number, small_number*tf.ones(tf.shape(dense)), dense)


    matrixTransformA = tf.matrix_band_part(tf.reshape(dense[:,:state_size*state_size],[-1,state_size,state_size]),0,-1)
    idx = state_size*state_size

    matrixDiagonalA = tf.matrix_diag(dense[:,idx:idx+state_size])
    idx += state_size

    matrixF = tf.reshape(dense[:,idx:idx+state_size*action_size],[-1,state_size,action_size])
    idx += state_size*action_size

    matrixGramQ = tf.reshape(dense[:,idx:idx+state_size*state_size],[-1,state_size,state_size])
    idx += state_size*state_size

    matrixGramR = tf.reshape(dense[:,idx:],[-1,action_size,action_size])

    matrixA = tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA),name="matrixA")
    matrixB = tf.matmul(matrixTransformA,matrixF,name="matrixB")
    matrixQ = tf.matmul(matrixGramQ,matrixGramQ,transpose_a=True,name="matrixQ")
    matrixR = tf.matmul(matrixGramR,matrixGramR,transpose_a=True,name="matrixR")

    matrixH11 = matrixA
    matrixH12 = tf.matmul(tf.matmul(-matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True)
    matrixH21 = -matrixQ
    matrixH22 = -tf.transpose(matrixA,perm=[0,2,1])
    matrixH1 = tf.concat((matrixH11,matrixH12),axis=2)
    matrixH2 = tf.concat((matrixH21,matrixH22),axis=2)
    matrixH = tf.concat((matrixH1,matrixH2),axis=1)
    eigVals, eigVecs = tf.self_adjoint_eig(matrixH)
    stableEigIdxs = tf.where(eigVals<0)
    er = tf.gather_nd(eigVecs,stableEigIdxs)
    # matrixV = eigVecs[:,tf.where(eigVals<0)[0]]
    # matrixV1 = matrixV[0:2][0:2]
    # matrixV2 = matrixV[2:4][0:2]
    # matrixP_ = tf.matmul(matrixV2,tf.matrix_inverse(matrixV1))
    # matrixK_ = tf.matmul(tf.matmul(tf.matrix_inverse(matrixR),matrixB,transpose_a=True),matrixP_)

    ###################################################################################################################3
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    idxs, v, e, er, da, A,B,R,Q= sess.run([stableEigIdxs, eigVecs, eigVals, er, matrixDiagonalA, matrixA, matrixB, matrixR, matrixQ, ], feed_dict={x: train_data[0:10]}) 
    # dense = sess.run([dense], feed_dict={x: train_data[0:2]}) 
    # A,B,Q,R,P_,K_ = sess.run([matrixA, matrixB, matrixR, matrixQ, matrixP_, matrixK_], feed_dict={x: train_data[0:10]}) 
    pdb.set_trace()

if __name__ == "__main__":
    tf.app.run()