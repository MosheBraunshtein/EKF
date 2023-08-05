import numpy as np
import matplotlib.pyplot as plt

k_range = np.arange(2,1001)
t = 0.01
noise_variance = 0.05

def H_k(k): 
    return np.array([[np.cos(k*t/2),np.cos(k*t),np.sin(k*t),np.sin(k*t/2)]])
def X_k(k):
    return 0.5*np.cos(k*t) + 0.1*np.sin(k*t/2) + np.random.normal(0, np.sqrt(noise_variance))

def compute_est_parameters():
    b = np.array([[0],[0],[0],[0]],dtype='float64')
    A = np.zeros((4,4),dtype='float64')
    est_a1 = np.array([])
    est_a2 = np.array([])
    est_a3 = np.array([])
    est_a4 = np.array([])
    for k in k_range:
        Hk = H_k(k)
        Xk = X_k(k)
        b += np.dot(Hk.T,Xk)
        A += np.dot(Hk.T,Hk)
        x = np.linalg.solve(A,b)
        est_a1 = np.append(est_a1,x[0])
        est_a2 = np.append(est_a2,x[1])
        est_a3 = np.append(est_a3,x[2])
        est_a4 = np.append(est_a4,x[3])
    return est_a1,est_a2,est_a3,est_a4

est_a1,est_a2,est_a3,est_a4 = compute_est_parameters()

plt.plot(k_range,est_a1,label='a1_estimate')
plt.plot(k_range,est_a2,label='a2_estimate')
plt.plot(k_range,est_a3,label='a3_estimate')
plt.plot(k_range,est_a4,label='a4_estimate')
plt.legend()
plt.show()


        