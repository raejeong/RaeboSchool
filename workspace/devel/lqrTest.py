import numpy as np
import scipy.linalg

network_out = 200*np.random.random(17)-100
A_ = np.array([network_out[:2],np.array([0,network_out[2]])])
D = np.diag(network_out[3:5])
F = np.array([network_out[5:7],network_out[7:9]])
Q_ = np.array([network_out[9:11],network_out[11:13]])
R_ = np.array([network_out[13:15],network_out[15:17]])

A = np.dot(np.dot(A_,D),np.linalg.inv(A_))
B = np.dot(A_,F)
Q = np.dot(Q_.T,Q_)
R = np.dot(R_.T,R_)

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

[K, P, eigVals] = lqr(A,B,Q,R)

H11 = A
H12 = np.dot(np.dot(-B,np.linalg.inv(R)),B.T)
H21 = -Q
H22 = -A.T
H1=np.concatenate((H11,H12),axis=1)
H2=np.concatenate((H21,H22),axis=1)
H = np.concatenate((H1,H2))
eigVals, eigVecs = np.linalg.eig(H)
V = eigVecs[:,np.where(eigVals<0)[0]]
V1 = V[0:2][0:2]
V2 = V[2:4][0:2]
P_ = np.dot(V2,np.linalg.inv(V1))
K_ = np.dot(np.dot(np.linalg.inv(R),B.T),P_)

print("P matrix error {}".format(np.mean(P-P_)))
print("K matrix error {}".format(np.mean(K-K_)))
