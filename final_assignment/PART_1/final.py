import numpy as np
import matplotlib.pyplot as plt

k_range = np.arange(1,1001)
t = 0.5
noise_variance = 0.05
def H_k(k): 
    return np.array([[np.cos(k*t/2),np.cos(k*t),np.sin(k*t),np.sin(k*t/2)]])
def X_k(k):
    return 0.5*np.cos(k*t) + 0.1*np.sin(k*t/2) + np.random.normal(0, np.sqrt(noise_variance))
def b():
    b = np.array([[0],[0],[0],[0]],dtype='float64')
    for k in k_range : 
        Hk = H_k(k)
        Xk = X_k(k)
        b += np.dot(Hk.T,Xk)
    return b
def A():
    A = np.zeros((4,4),dtype='float64')
    for k in k_range : 
        Hk = H_k(k)
        A += np.dot(Hk.T,Hk)
    return A
def plotX():
    x_k = []
    for k in k_range:
        xk = X_k(k)
        x_k.append(xk)
    plt.plot(x_k)
    plt.show()

A = A()
b = b()
x = np.linalg.solve(A,b)
plotX()
print(x)


        