import numpy as np
import math
import constants

# constants for the kalman filter 
thymio_speed_to_mms = 0.3
thymio_speed_to_rads = 0.007
q_rot = 0.0001
r_rot = 0.0001
q_nu = 0.04
r_nu = 0.04
qp = 0.25 
rp = 0.25  
q_alpha = (5.2 * 10**(-5))/2
r_alpha = (5.2 * 10**(-5))/2

def get_angle(direction):
    
    alpha = math.atan2(direction[1], direction[0])
    
    return alpha
    

def get_vector_from_angle(angle):

    x = math.cos(angle)  # coordinate x
    y = math.sin(angle)  # coordinate y
    return (x, y)

def compute_jacobian(x_est, Ts):

    _, _, v, theta, _ = x_est.flatten()

    A = np.array([
        [1, 0, np.cos(theta) * Ts, -v * np.sin(theta) * Ts, 0],  
        [0, 1, np.sin(theta) * Ts,  v * np.cos(theta) * Ts, 0], 
        [0, 0, 1,                  0,                       0], 
        [0, 0, 0,                  1,                       Ts], 
        [0, 0, 0,                  0,                       1]  
    ])

    return A


def motion_model(x_est, Ts):
 
    x, y, v, theta, omega = x_est.flatten()  # state variables
    
    # next state
    x_next = np.array([
        [x + v * np.cos(theta) * Ts],
        [y + v * np.sin(theta) * Ts],
        [v],
        [theta + omega * Ts],
        [omega]
    ])
    return x_next

def get_avg_speed_and_omega(motor_speed_left, motor_speed_right):
    
    turning_left = False
    turning_right = False
    going_straight = False
    pivoting_left = False 
    pivoting_right = False 
       
    if (motor_speed_left > motor_speed_right):
        turning_right = True
        turning_left = False
        
    elif (motor_speed_left < motor_speed_right):
        turning_left = True
        turning_right = False
    
    elif (motor_speed_left == motor_speed_right):
        going_straight = True
        
    elif (-motor_speed_left == motor_speed_right and motor_speed_left<0):
        pivoting_left = True
        
    elif (motor_speed_left == -motor_speed_right and motor_speed_right<0):
        pivoting_right = True
    
    
    if turning_left:
        avg_omega = -(np.abs(motor_speed_left) + np.abs(motor_speed_right))/2
        avg_speed = (motor_speed_left + motor_speed_right)/2
    
    elif turning_right:
        avg_omega = (np.abs(motor_speed_left) + np.abs(motor_speed_right))/2
        avg_speed = (motor_speed_left + motor_speed_right)/2
        
    elif going_straight:
        avg_omega = 0
        avg_speed = (motor_speed_left + motor_speed_right)/2
        
    elif pivoting_left:
        avg_omega = -(np.abs(motor_speed_left) + np.abs(motor_speed_right))/2
        avg_speed = 0
        
    elif pivoting_right:
        avg_omega = (np.abs(motor_speed_left) + np.abs(motor_speed_right))/2
        avg_speed = 0



    return avg_speed, avg_omega
        

    


def extended_kalman_filter(motor_speed_left, motor_speed_right, pos, alpha, x_est_prev, P_est_prev, camera):
        
    speed, omega = get_avg_speed_and_omega(motor_speed_left, motor_speed_right)
    
    Q = np.array([[qp, 0, 0, 0,0], [0, qp,0 ,0,0], [0,0, q_nu, 0,0], [0,0,0,q_alpha,0],[0,0,0,0,q_rot]])
    
    # estimated state
    x_est_a_priori = motion_model(x_est_prev, constants.Ts)
    A = compute_jacobian(x_est_prev, constants.Ts)
    
    # estimated covariance
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori
    
    # measurements         
    if (camera) : 
        y = np.array([[pos[0]],[pos[1]],[speed*thymio_speed_to_mms],[alpha],[omega*thymio_speed_to_rads]])
        H = np.eye(5)  
        R = np.diag([rp, rp, r_nu, r_alpha, r_rot])  
    else:
        # no camera (=no position), use only the speed
        y = np.array([[speed * thymio_speed_to_mms],[omega*thymio_speed_to_rads]]) 
        H = np.array([[0, 0, 1, 0, 0],  
                    [0, 0, 0, 0, 1]]) 
        R = np.diag([r_nu, r_rot])  

    i = y - np.dot(H, x_est_a_priori)
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
             
    # kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))
    
    
    # a posteriori estimate
    x_est = x_est_a_priori + np.dot(K,i)
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori))
     
    return x_est, P_est