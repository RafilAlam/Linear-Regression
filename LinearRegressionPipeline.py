import torch
import torch.nn as nn

'''
 Linear Regression Model
 
 f = w * x + b

 where f = function output, x = input, w = weight, b = bias

 Using the simple built-in model from PyTorch:
 
 model = nn.Linear(input_size, output_size)
'''

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define different layers
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

w = torch.tensor(0.0, requires_grad = True)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size, output_size = n_features, n_features

learning_rate = 0.01
n_epochs = 100

model = LinearRegression(input_size, output_size)
loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f'Prediction before training: f({X_test}) = {model(X_test).item():.3f}')

for epoch in range(n_epochs):

    # predict = forward pass (no need .forward() since it is the only function in the class)
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()

    # update weights
    optimiser.step()

    # zero the gradients after updating
    optimiser.zero_grad()

    if (epoch+1) % 10 == 0:
        w, b = model.parameters() # unpack parameters
        print(f'epoch {epoch + 1}: w = {w[0][0].item()}, loss = {l.item()}')

print(f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')