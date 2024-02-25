import copy
import numpy as np

def compute_cost(x,y,w,b):
    m = x.shape[0]
    loss = 0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        loss += (f_wb - y[i]) ** 2
    loss = loss / (2*m)
    return loss

def compute_gradient(x,y,w,b):
    m,n = x.shape
    dj_w = np.zeros((n))
    dj_b = 0
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_w[j] = dj_w[j] + err * x[i, j]
        dj_b += err
    dj_w /= m
    dj_b /= m
    
    return dj_w, dj_b

def gradient_descent(x,y,w_in,b_in,alpha,epochs):
    w = copy.deepcopy(w_in) # avoid modifying global w within function
    b = b_in
    
    for i in range(epochs):
        dj_w, dj_b = compute_gradient(x,y,w,b)
        
        # Update
        w = w - alpha * dj_w
        b = b - alpha * dj_b
        
        if i % 1000 == 0:
            cost = compute_cost(x, y, w, b)
            print("Iteration:", i, " Cost:", cost)
    return w,b

def predict(x,w,b):
    f_wb = []
    for i in range(x.shape[0]):
        f_wb.append(np.dot(x[i], w) + b)
    return np.array(f_wb)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

