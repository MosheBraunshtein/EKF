import numpy as np
import matplotlib.pyplot as plt
from animation_plot import AnimationPlot

# Parameters
mean_x0 = np.zeros(3)
P_k_initial = np.array([[25, 0, 0], [0, 25, 0], [0, 0, 0.1]])
P_k = np.array([[25, 0, 0], [0, 25, 0], [0, 0, 0.1]])
mean_w = np.zeros(3)
Q = np.array([[0.01,0,0],[0,0.01,0],[0,0,0]])
Q_tuned = np.array([[0.01,0,0],[0,0.01,0],[0,0,0]])
mean_v = np.zeros(2)
R = np.array([[0.01,0],[0,0.1]])
R_tuned = np.array([[0.01,0],[0,0.01]])
num_iterations = 300
t = 0.1  
n = 0.05

# initialization
X = np.random.multivariate_normal(mean_x0, P_k_initial)
X_0_reference = np.array([[0],[X[1]],[X[2]]])
X_0_estimate = X_0_reference

# array to store values
X_k_reference_values = [X_0_reference]
X_k_estimate_values = [X_0_estimate] 
P_k_values = [P_k]
P_k_trace_values = [np.trace(P_k)]
Z_k_values = []
def Fk(X3_kplus1_k):
    F_13 = -t*np.sin(X3_kplus1_k)
    F_23 = t*np.cos(X3_kplus1_k)
    return np.array([[1,0,F_13],[0,1,F_23],[0,0,1]])
def Hk(X1_kplus1_k,X2_kplus1_k):
    H_11 = X1_kplus1_k/np.sqrt(((X1_kplus1_k**2)+(X2_kplus1_k**2)))
    H_12 = X2_kplus1_k/np.sqrt(((X1_kplus1_k**2)+(X2_kplus1_k**2)))
    H_21 = -X2_kplus1_k/((X1_kplus1_k**2)+(X2_kplus1_k**2))
    H_22 = X1_kplus1_k/((X1_kplus1_k**2)+(X2_kplus1_k**2))
    return np.array([[H_11,H_12,0],[H_21,H_22,0]])
def P_prediction_step(F_k,P_k):
    return np.dot(F_k,np.dot(P_k,F_k.T)) + Q_tuned
def h(X1_kplus1_k,X2_kplus1_k):
    z11 = np.sqrt((X1_kplus1_k**2)+(X2_kplus1_k**2))
    z21 = np.arctan2(X2_kplus1_k,X1_kplus1_k)
    Zk_excepted = np.array([[z11],[z21]]) 
    return Zk_excepted
def new_measurement(X1_kplus1,X2_kplus1,vk):
    z11 = np.sqrt((X1_kplus1)**2+(X2_kplus1)**2)
    z21 = np.arctan2(X2_kplus1,X1_kplus1)
    Zk = np.array([[z11]+vk[0],[z21]+vk[1]/(180/np.pi)]) 
    return Zk
def kalman_gain(H_kplus1,P_kplus1_k):
    residual_covariance_matrix = np.dot(H_kplus1,np.dot(P_kplus1_k,H_kplus1.T)) + R_tuned
    residual_covariance_matrix_inv = np.linalg.inv(residual_covariance_matrix)
    return np.dot(np.dot(P_kplus1_k,H_kplus1.T),residual_covariance_matrix_inv)
def state_update_with_measurement(X_kplus1_k,Z_excepted,Z_kplus1,kalman_gain_kplus1):
    innovation_k = Z_kplus1-Z_excepted
    return X_kplus1_k + np.dot(kalman_gain_kplus1,innovation_k)
def covariance_update(kalman_gain_kplus1,H_kplus1,P_kplus1_k):
    d = np.eye(3)-np.dot(kalman_gain_kplus1,H_kplus1)
    return np.dot(d,np.dot(P_kplus1_k,d.T)) + np.dot(kalman_gain_kplus1,np.dot(R,kalman_gain_kplus1.T))


for _ in range(num_iterations):
    Wk = np.random.multivariate_normal(np.zeros(3), Q)
    vk = np.random.multivariate_normal(np.zeros(2), R)
    #compute reference trajectory 
    X_k_reference = X_k_reference_values[-1]
    X1_kplus1_reference = X_k_reference[0,0] + np.cos(X_k_reference[2,0]) * t 
    X2_kplus1_reference = X_k_reference[1,0] + np.sin(X_k_reference[2,0]) * t 
    X3_kplus1_reference = X_k_reference[2,0] + n 
    X_kplus1_reference = np.array([[X1_kplus1_reference],[X2_kplus1_reference],[X3_kplus1_reference]])
    X_k_reference_values.append(X_kplus1_reference)
    # generate observations and estimate X
    X_k_k_estimate = X_k_estimate_values[-1]
    X1_kplus1_k = X_k_k_estimate[0,0] + np.cos(X_k_k_estimate[2,0]) * t + Wk[0]
    X2_kplus1_k = X_k_k_estimate[1,0] + np.sin(X_k_k_estimate[2,0]) * t + Wk[1]
    X3_kplus1_k = X_k_k_estimate[2,0] + n + Wk[2]
    X_kplus1_k = np.array([[X1_kplus1_k],[X2_kplus1_k],[X3_kplus1_k]])
    F_k = Fk(X3_kplus1_k)
    P_k = P_k_values[-1]
    P_kplus1_k = P_prediction_step(F_k,P_k)
    H_kplus1 = Hk(X1_kplus1_k,X2_kplus1_k)
    kalman_gain_kplus1 = kalman_gain(H_kplus1,P_kplus1_k)
    Z_kplus1 = new_measurement(X1_kplus1_reference,X2_kplus1_reference,vk)
    Z_k_values.append(Z_kplus1)
    Zk_excepted = h(X1_kplus1_k,X2_kplus1_k)
    X_kplus1_kplus1 = state_update_with_measurement(X_kplus1_k,Zk_excepted,Z_kplus1,kalman_gain_kplus1)
    X_k_estimate_values.append(X_kplus1_kplus1)
    P_kplus1_kplus1 = covariance_update(kalman_gain_kplus1,H_kplus1,P_kplus1_k)
    # if(np.trace(P_kplus1_kplus1)<1):
    #     P_kplus1_kplus1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.05]])
    P_k_values.append(P_kplus1_kplus1)
    P_k_trace_values.append(np.trace(P_kplus1_kplus1))
    # R_tuned  = R
    # if(X_k_k_estimate[2,0]<0.6):
    #     #listen to measurements
    #     R_tuned = np.array([[0.007,0],[0,0.07]])
    
        


x1_reference = []
x2_reference = []
x1_estimate = []
x2_estimate = []
z1 = []
z2 = []

for element in X_k_reference_values:
    x1_reference.append(element[0])
    x2_reference.append(element[1])
for element in X_k_estimate_values:
    x1_estimate.append(element[0])
    x2_estimate.append(element[1])
for element in Z_k_values:
    r = element[0]
    #rad
    teta = element[1] 
    z1.append(r*np.cos(teta))
    z2.append(r*np.sin(teta))
plt.plot(x1_reference, x2_reference,'red')
plt.plot(x1_estimate, x2_estimate,color='green')
plt.plot(z1, z2, "o",markersize=1)
plt.figure()
plt.plot(P_k_trace_values)
plt.show()


# pl = AnimationPlot(x1_reference,x2_reference,X_k_estimate_values,Z_k_values)
# pl.plot()












