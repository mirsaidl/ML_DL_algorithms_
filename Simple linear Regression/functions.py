import numpy

# Compute Gradient
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_w = 0
    dj_b = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_w += (f_wb - y[i]) * x[i]  # Corrected the calculation of dj_w
        dj_b += (f_wb - y[i])          # Corrected the calculation of dj_b
    dj_w /= m
    dj_b /= m

    return dj_w, dj_b

# Compute cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    return 1 / (2 * m) * cost

# Gradient Descent
def gradient_descent(x, y, w_in, b_in, alpha, iterations):
    b = b_in
    w = w_in
    for _ in range(iterations):
        dj_w, dj_b = compute_gradient(x, y, w, b)

        b = b - alpha * dj_b  # Corrected the update for b
        w = w - alpha * dj_w  # Corrected the update for w

        # Print the cost every few iterations
        if _ % 1000 == 0:
            cost = compute_cost(x, y, w, b)
            print("Iteration:", _, " Cost:", cost)

    return w, b

def predict(x,w,b):
    m = x.shape[0]
    y = []
    for i in range(m):
        y.append(w * x[i] + b)
    return np.array(y)

# Calculate MAE
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
mae(y_test, y_predicted)