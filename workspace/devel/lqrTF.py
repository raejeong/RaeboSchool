import numpy as np 
import tensorflow as tf 
import scipy.linalg
import pdb


tf.logging.set_verbosity(tf.logging.INFO)

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals


def main(unused_argv):
    t = tf.float64
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array

    small_number = 0.001
    state_size = 3
    action_size = 2
    output_size = state_size + action_size**2 + state_size*action_size + state_size**2

    x = tf.placeholder(t, [None, 784])

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
    dense = tf.where(tf.logical_and((dense >= 0),(dense<small_number)), small_number*tf.ones(tf.shape(dense),t), dense)
    dense = tf.where(tf.logical_and((dense < 0),(tf.abs(dense)<small_number)), -small_number*tf.ones(tf.shape(dense),t), dense)


    matrixTransformA = tf.matrix_band_part(tf.reshape(dense[:,:state_size*state_size],[-1,state_size,state_size]),0,-1)
    idx = state_size*state_size

    matrixDiagonalA = tf.matrix_diag(dense[:,idx:idx+state_size])
    idx += state_size

    matrixF = tf.matrix_band_part(tf.reshape(dense[:,idx:idx+state_size*action_size],[-1,state_size,action_size]),action_size-1,-1)
    idx += state_size*action_size

    matrixGramR = tf.reshape(dense[:,idx:],[-1,action_size,action_size])

    matrixGramA = tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA))
    
    matrixA = 10*tf.matmul(matrixGramA,matrixGramA,transpose_a=True,name="matrixA")
    matrixB = 10*tf.matmul(matrixTransformA,matrixF,name="matrixB")
    matrixR = 10*tf.matmul(matrixGramR,matrixGramR,transpose_a=True,name="matrixR")
    matrixQ = tf.transpose(tf.matmul(tf.matmul(matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True),perm=[0,2,1])

    matrixH11 = matrixA
    matrixH12 = tf.matmul(tf.matmul(-matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True)
    matrixH21 = -matrixQ
    matrixH22 = -tf.transpose(matrixA,perm=[0,2,1])
    matrixH1 = tf.concat((matrixH11,matrixH12),axis=2)
    matrixH2 = tf.concat((matrixH21,matrixH22),axis=2)
    matrixH = tf.concat((matrixH1,matrixH2),axis=1)
    eigVals, eigVecs = tf.self_adjoint_eig(matrixH)
    eigVecs = tf.transpose(eigVecs,perm=[0,2,1])
    stableEigIdxs = tf.where(eigVals<0)
    matrixV = tf.reshape(tf.gather_nd(eigVecs,stableEigIdxs),[-1,state_size,2*state_size])
    matrixV1 = tf.transpose(matrixV[:,:,:state_size],perm=[0,2,1])
    matrixV2 = tf.transpose(matrixV[:,:,state_size:],perm=[0,2,1])
    matrixP_ = tf.matmul(matrixV2,tf.matrix_inverse(matrixV1))
    matrixK_ = tf.matmul(tf.matmul(tf.matrix_inverse(matrixR),matrixB,transpose_b=True),matrixP_)

    ###################################################################################################################3
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_size = 100

    # idxs, v, e, da, A,B,R,Q,H,V,V1,V2= sess.run([stableEigIdxs, eigVecs, eigVals, matrixDiagonalA, matrixA, matrixB, matrixR, matrixQ, matrixH,matrixV,matrixV1,matrixV2], feed_dict={x: train_data[0:2]}) 
    # pdb.set_trace()
    # dense = sess.run([dense], feed_dict={x: train_data[0:2]}) 
    A,B,Q,R,P_,K_,e= sess.run([matrixA, matrixB, matrixQ, matrixR, matrixP_, matrixK_, eigVals], feed_dict={x: train_data[0:batch_size]})

    P_error = []
    K_error = []
    for i in range(batch_size):
        a = A[i]
        b = B[i]
        q = Q[i]
        r = R[i]
        [K, P, _] = lqr(a,b,q,r)
        P_error.append(P-P_[i])
        K_error.append(K-K_[i])
        
    print("P matrix error {}".format(np.mean(P_error)))
    print("K matrix error {}".format(np.mean(K_error)))

if __name__ == "__main__":
    tf.app.run()