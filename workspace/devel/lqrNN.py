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
    logs_path="/home/rae/RaeboSchool/workspace/devel/log/"
    t = tf.float32
    small_number = 0.0001

    state_size = 15
    action_size = 3
    input_size = 2*state_size**2 + state_size + state_size*action_size + action_size**2
    output_size = state_size*action_size

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.3)

    x = tf.placeholder(t, [None, input_size])
    # x = tf.where(tf.logical_and((x >= 0),(x<small_number)), small_number*tf.ones(tf.shape(x),t), x)
    # x = tf.where(tf.logical_and((x < 0),(tf.abs(x)<small_number)), -small_number*tf.ones(tf.shape(x),t), x)
    y = tf.placeholder(t, [None, action_size, state_size])

    fc1 = tf.contrib.layers.layer_norm(tf.layers.dense(inputs=x, units=input_size, name="lqrfc1", activation=tf.nn.tanh, kernel_regularizer=regularizer)) - x
    fc2 = tf.contrib.layers.layer_norm(tf.layers.dense(inputs=fc1, units=input_size, name="lqrfc2", activation=tf.nn.tanh, kernel_regularizer=regularizer)) - fc1
    fc3 = tf.contrib.layers.layer_norm(tf.layers.dense(inputs=fc2, units=input_size, name="lqrfc3", activation=tf.nn.tanh, kernel_regularizer=regularizer)) - fc2
    fc4 = tf.contrib.layers.layer_norm(tf.layers.dense(inputs=fc3, units=input_size, name="lqrfc4", activation=tf.nn.tanh, kernel_regularizer=regularizer)) - fc3
    fc5 = tf.contrib.layers.layer_norm(tf.layers.dense(inputs=fc4, units=input_size, name="lqrfc5", activation=tf.nn.tanh, kernel_regularizer=regularizer)) - fc4

    lqr_out = tf.reshape(tf.layers.dense(inputs=fc5, units=output_size, name="lqrout", kernel_regularizer=regularizer),[-1,action_size,state_size])
    
    matrixTransformA = tf.reshape(x[:,:state_size*state_size],[-1,state_size,state_size])
    idx = state_size*state_size
    matrixDiagonalA = tf.matrix_diag(x[:,idx:idx+state_size])
    idx += state_size
    matrixF = tf.reshape(x[:,idx:idx+state_size*action_size],[-1,state_size,action_size])
    idx += state_size*action_size
    matrixGramQ = tf.reshape(x[:,idx:idx+state_size**2],[-1,state_size,state_size])
    idx += state_size**2
    matrixGramR = tf.reshape(x[:,idx:idx+action_size**2],[-1,action_size,action_size])

    matrixA = tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA),name="matrixA")
    matrixB = tf.matmul(matrixTransformA,matrixF,name="matrixB")/tf.abs(tf.reduce_mean(matrixF)*state_size)

    matrixQ = tf.matmul(matrixGramQ,matrixGramQ,transpose_a=True)/tf.abs(tf.reduce_mean(matrixGramQ)*state_size)
    matrixQU = tf.matrix_band_part(matrixQ,0,-1)
    matrixQ = matrixQU + tf.transpose(matrixQU,perm=[0,2,1]) - tf.matrix_band_part(matrixQ,0,0)

    matrixR = tf.matmul(matrixGramR,matrixGramR,transpose_a=True)/tf.abs(tf.reduce_mean(matrixGramR)*action_size)
    matrixRU = tf.matrix_band_part(matrixR,0,-1)
    matrixR = matrixRU + tf.transpose(matrixRU,perm=[0,2,1]) - tf.matrix_band_part(matrixR,0,0)

    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss = tf.reduce_mean(tf.losses.huber_loss(y, lqr_out)) + reg_term
    lqr_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_lqr_net = lqr_optimizer.minimize(loss)

    summary = tf.summary.scalar("loss", loss)

    ###################################################################################################################
    
    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter(logs_path, sess.graph)
    tf.global_variables_initializer().run()

    iterations = 10000000
    batch_size = 500
    for i in range(iterations):
        random_data = np.random.uniform(-1,1,size=[3*batch_size, input_size])
        label = []
        train_data = []
        batch_count = 0
        j = 0
        A,B,Q,R,TA,DA,F,K = sess.run([matrixA, matrixB, matrixQ, matrixR, matrixTransformA, matrixDiagonalA, matrixF,lqr_out], feed_dict={x: random_data})
        while batch_count < batch_size:
            j += 1
            try:
                label.append(control.lqr(A[j],B[j],Q[j],R[j])[0])
                train_data.append(random_data[j])
                batch_count += 1
            except:
                continue
        label = np.array(label)
        train_data = np.array(train_data)
        lqr_loss, loss_summary, _ = sess.run([loss, summary, train_lqr_net],feed_dict={x:train_data,y:label})
        if lqr_loss < 1e5:
            writer.add_summary(loss_summary, i)
        if i%100==0:
            print("loss: {}".format(lqr_loss))
            print(label[0]-K[0])

    writer.close()
    pdb.set_trace()

if __name__ == "__main__":
    tf.app.run()

