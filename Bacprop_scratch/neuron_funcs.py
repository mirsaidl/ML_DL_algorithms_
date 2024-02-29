import numpy as np




# Normalize data
def normalize_data(x):
    mean = np.mean(x)
    st_dev = np.std(x)
    xn = (x - mean) / st_dev
    return xn

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define forward propagation
def forward_propagation(X, W1, W2, b1, b2):
    # Input to hidden layer
    hidden_input = np.dot(X, W1) + b1
    # Apply activation function (sigmoid) to hidden layer
    hidden_output = sigmoid(hidden_input)
    
    # Hidden layer to output layer
    output = np.dot(hidden_output, W2) + b2
    # Apply activation function (sigmoid) to output layer
    output = sigmoid(output)
    
    return hidden_output, output

# Define backpropagation
def backpropagation(X, y, hidden_output, output, W1, W2, b1, b2, learning_rate):
    # Compute error at output layer
    output_error = y - output
    # Compute gradient at output layer
    output_delta = output_error * sigmoid_derivative(output)
    
    # Compute error at hidden layer
    hidden_error = np.dot(output_delta, W2)
    # Compute gradient at hidden layer
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    W2 += np.dot(hidden_output.T, output_delta) * learning_rate
    W1 += np.dot(X.T, hidden_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0) * learning_rate
    b1 += np.sum(hidden_delta, axis=0) * learning_rate
    
    return W1, W2, b1, b2

# Implement large neural network with multiclass
def fit(X, y, hidden_size, num_classes, learning_rate, epochs):
    # Initialize weights and biases
    input_size = X.shape[1]
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, num_classes)
    b2 = np.zeros((1, num_classes))

    # Training loop
    for epoch in range(epochs):
        # Forward propagation
        hidden_output, output = forward_propagation(X, W1, W2, b1, b2)
        
        # Backpropagation
        W1, W2, b1, b2 = backpropagation(X, y, hidden_output, output, W1, W2, b1, b2, learning_rate)
        
        # Print loss (optional)
        if epoch % 100 == 0:
            loss = sparse_categorical_crossentropy(y, output)
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return W1, W2, b1, b2

# Make predictions using trained weights
def predict(X, W1, W2, b1, b2):
    hidden_output, output = forward_propagation(X, W1, W2, b1, b2)
    yhat = np.argmax(output, axis=1)  # Choose class with highest probability
    return yhat