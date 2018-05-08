import numpy as np 
import tensorflow as tf 
import scipy.linalg
import control
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

    small_number = 0.0000001
    state_size = 3
    action_size = 2
    output_size = state_size + state_size*action_size + 2*state_size**2

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
    dense = dense/tf.reduce_mean(dense)

    matrixTransformA = tf.matrix_band_part(tf.reshape(dense[:,:state_size*state_size],[-1,state_size,state_size]),0,-1)
    idx = state_size*state_size

    matrixDiagonalA = tf.matrix_diag(dense[:,idx:idx+state_size])

    idx += state_size

    matrixF = tf.reshape(dense[:,idx:idx+state_size*action_size],[-1,state_size,action_size])/(state_size*action_size)
    idx += state_size*action_size

    matrixGramQ = tf.reshape(dense[:,idx:],[-1,state_size,state_size])/(state_size*state_size)

    matrixGramA = tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA))/(state_size**2)

    matrixA = tf.matmul(matrixGramA,matrixGramA,transpose_a=True)/(state_size**2)
    matrixAU = tf.matrix_band_part(matrixA,0,-1)
    matrixB = tf.matmul(matrixTransformA,matrixF,name="matrixB")
    matrixQ = tf.matmul(matrixGramQ,matrixGramQ,transpose_a=True)/(state_size**2)
    matrixQU = tf.matrix_band_part(matrixQ,0,-1)

    matrixA = matrixAU + tf.transpose(matrixAU,perm=[0,2,1]) - tf.matrix_band_part(matrixA,0,0)
    matrixQ = matrixQU + tf.transpose(matrixQU,perm=[0,2,1]) - tf.matrix_band_part(matrixQ,0,0)
    matrixR = tf.matmul(tf.matmul(matrixB,tf.matrix_inverse(matrixQ),transpose_a=True),matrixB)
    matrixQ = tf.transpose(tf.matmul(tf.matmul(matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True),perm=[0,2,1])

    matrixH11 = matrixA
    matrixH12 = -matrixQ
    matrixH21 = -matrixQ
    matrixH22 = -matrixA
    matrixH1 = tf.concat((matrixH11,matrixH12),axis=2)
    matrixH2 = tf.concat((matrixH21,matrixH22),axis=2)
    matrixH = tf.concat((matrixH1,matrixH2),axis=1)
    # matrixH = tf.where(tf.logical_and((matrixH >= 0),(matrixH<small_number)), small_number*tf.ones(tf.shape(matrixH),t), matrixH)
    # matrixH = tf.where(tf.logical_and((matrixH < 0),(tf.abs(matrixH)<small_number)), -small_number*tf.ones(tf.shape(matrixH),t), matrixH)
    matrixHU = tf.matrix_band_part(matrixH,0,-1)
    matrixH = matrixHU + tf.transpose(matrixHU,perm=[0,2,1]) - tf.matrix_band_part(matrixH,0,0)

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

    batch_size = 2

    # idxs, v, e, da, A,B,R,Q,H, at,ag= sess.run([stableEigIdxs, eigVecs, eigVals, matrixDiagonalA, matrixA, matrixB, matrixR, matrixQ, matrixH,matrixTransformA,matrixGramA], feed_dict={x: train_data[0:batch_size]}) 
    # pdb.set_trace()
    # idxs, v, e, da, A,B,R,Q,H,V,V1,V2= sess.run([stableEigIdxs, eigVecs, eigVals, matrixDiagonalA, matrixA, matrixB, matrixR, matrixQ, matrixH,matrixV,matrixV1,matrixV2], feed_dict={x: train_data[0:batch_size]}) 
    # # dense = sess.run([dense], feed_dict={x: train_data[0:2]}) 
    A,B,Q,R,P_,K_,e,H,AG= sess.run([matrixA, matrixB, matrixQ, matrixR, matrixP_, matrixK_, eigVals, matrixH,matrixGramA], feed_dict={x: train_data[0:batch_size]})
    # pdb.set_trace()

    P_error = []
    K_error = []
    for i in range(batch_size):
        eigVals, eigVecs = np.linalg.eig(H[i])
        V = eigVecs[:,np.where(eigVals<0)[0]]
        V1 = V[:state_size]
        V2 = V[state_size:]
        P = np.dot(V2,np.linalg.inv(V1))
        K = np.dot(np.dot(np.linalg.inv(R[i]),B[i].T),P)
        P_error.append(np.abs(P-P_[i]))
        K_error.append(np.abs(K-K_[i]))
        # pdb.set_trace()
        
    print("P matrix error {}".format(np.mean(P_error)))
    print("K matrix error {}".format(np.mean(K_error)))
    pdb.set_trace()


if __name__ == "__main__":
    tf.app.run()




# tf.logging.set_verbosity(tf.logging.INFO)

# def lqr(A,B,Q,R):
#     """Solve the continuous time lqr controller.
     
#     dx/dt = A x + B u
     
#     cost = integral x.T*Q*x + u.T*R*u
#     """
#     #ref Bertsekas, p.151
 
#     #first, try to solve the ricatti equation
#     X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
#     #compute the LQR gain
#     K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
#     eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
#     return K, X, eigVals


# def main(unused_argv):
#     t = tf.float64
#     # Load training and eval data
#     mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#     train_data = mnist.train.images # Returns np.array

#     small_number = 0.0000001
#     state_size = 3
#     action_size = 2
#     output_size = state_size + action_size**2 + state_size*action_size + state_size**2

#     x = tf.placeholder(t, [None, 784])

#     input_layer = tf.reshape(x,[-1,28,28,1])

#     conv1 = tf.layers.conv2d(
#       inputs=input_layer,
#       filters=32,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)

#     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#     conv2 = tf.layers.conv2d(
#       inputs=pool1,
#       filters=64,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)
#     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
#     dense = tf.layers.dense(inputs=pool2_flat, units=output_size)
#     dense = dense/tf.reduce_mean(dense)

#     matrixTransformA = tf.matrix_band_part(tf.reshape(dense[:,:state_size*state_size],[-1,state_size,state_size]),0,-1)
#     idx = state_size*state_size

#     matrixDiagonalA = tf.matrix_diag(dense[:,idx:idx+state_size])

#     idx += state_size

#     matrixF = tf.reshape(dense[:,idx:idx+state_size*action_size],[-1,state_size,action_size])/(state_size*action_size)
#     idx += state_size*action_size

#     matrixGramR = tf.reshape(dense[:,idx:],[-1,action_size,action_size])/(state_size*action_size)

#     matrixGramA = tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA))/(state_size**2)

#     matrixAU = tf.matrix_band_part(tf.matmul(matrixGramA,matrixGramA,transpose_a=True,name="matrixA")/(state_size**2),0,-1)
#     matrixB = tf.matmul(matrixTransformA,matrixF,name="matrixB")
#     matrixRU = tf.matrix_band_part(tf.matmul(matrixGramR,matrixGramR,transpose_a=True,name="matrixR"),0,-1)

#     matrixA = 0.5*(matrixAU + tf.transpose(matrixAU,perm=[0,2,1]))
#     matrixR = 0.5*(matrixRU + tf.transpose(matrixRU,perm=[0,2,1]))

#     # matrixQU = tf.matrix_band_part(tf.transpose(tf.matmul(tf.matmul(matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True),perm=[0,2,1]),0,-1)
#     matrixQ = tf.transpose(tf.matmul(tf.matmul(matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True),perm=[0,2,1])
#     # matrixQ = 0.5*(matrixQU + tf.transpose(matrixQU,perm=[0,2,1]))


#     matrixH11 = matrixA
#     matrixH12 = -matrixQ
#     matrixH21 = -matrixQ
#     matrixH22 = -tf.transpose(matrixA,perm=[0,2,1])
#     matrixH1 = tf.concat((matrixH11,matrixH12),axis=2)
#     matrixH2 = tf.concat((matrixH21,matrixH22),axis=2)
#     matrixH = tf.concat((matrixH1,matrixH2),axis=1)
#     matrixH = tf.where(tf.logical_and((matrixH >= 0),(matrixH<small_number)), small_number*tf.ones(tf.shape(matrixH),t), matrixH)
#     matrixH = tf.where(tf.logical_and((matrixH < 0),(tf.abs(matrixH)<small_number)), -small_number*tf.ones(tf.shape(matrixH),t), matrixH)
#     matrixHU = tf.matrix_band_part(matrixH,0,-1)
#     matrixH = matrixHU + tf.transpose(matrixHU,perm=[0,2,1]) -tf.matrix_band_part(matrixH,0,0)


#     eigVals, eigVecs = tf.self_adjoint_eig(matrixH)
#     eigVecs = tf.transpose(eigVecs,perm=[0,2,1])
#     stableEigIdxs = tf.where(eigVals<0)
#     matrixV = tf.reshape(tf.gather_nd(eigVecs,stableEigIdxs),[-1,state_size,2*state_size])
#     matrixV1 = tf.transpose(matrixV[:,:,:state_size],perm=[0,2,1])
#     matrixV2 = tf.transpose(matrixV[:,:,state_size:],perm=[0,2,1])
#     matrixP_ = tf.matmul(matrixV2,tf.matrix_inverse(matrixV1))
#     matrixK_ = tf.matmul(tf.matmul(tf.matrix_inverse(matrixR),matrixB,transpose_b=True),matrixP_)

#     ###################################################################################################################3
    
#     sess = tf.InteractiveSession()
#     tf.global_variables_initializer().run()

#     batch_size = 2

#     # idxs, v, e, da, A,B,R,Q,H, at,ag= sess.run([stableEigIdxs, eigVecs, eigVals, matrixDiagonalA, matrixA, matrixB, matrixR, matrixQ, matrixH,matrixTransformA,matrixGramA], feed_dict={x: train_data[0:batch_size]}) 
#     # pdb.set_trace()
#     # idxs, v, e, da, A,B,R,Q,H,V,V1,V2= sess.run([stableEigIdxs, eigVecs, eigVals, matrixDiagonalA, matrixA, matrixB, matrixR, matrixQ, matrixH,matrixV,matrixV1,matrixV2], feed_dict={x: train_data[0:batch_size]}) 
#     # # dense = sess.run([dense], feed_dict={x: train_data[0:2]}) 
#     A,B,Q,R,P_,K_,e,H= sess.run([matrixA, matrixB, matrixQ, matrixR, matrixP_, matrixK_, eigVals, matrixH], feed_dict={x: train_data[0:batch_size]})
#     # pdb.set_trace()

#     P_error = []
#     K_error = []
#     for i in range(batch_size):
#         eigVals, eigVecs = np.linalg.eig(H[i])
#         V = eigVecs[:,np.where(eigVals<0)[0]]
#         V1 = V[:state_size]
#         V2 = V[state_size:]
#         P = np.dot(V2,np.linalg.inv(V1))
#         K = np.dot(np.dot(np.linalg.inv(R[i]),B[i].T),P)
#         P_error.append(np.abs(P-P_[i]))
#         K_error.append(np.abs(K-K_[i]))
#         # pdb.set_trace()
        
#     print("P matrix error {}".format(np.mean(P_error)))
#     print("K matrix error {}".format(np.mean(K_error)))
#     pdb.set_trace()


# if __name__ == "__main__":
#     tf.app.run()