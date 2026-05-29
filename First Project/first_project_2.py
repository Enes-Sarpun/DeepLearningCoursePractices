## Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

## Data:
data = pd.read_csv("study_hours_grades.csv")

X = torch.tensor(data['study_hours'].values, dtype=torch.float32).unsqueeze(1) # Shape (100,) -> (100, 1)
y = torch.tensor(data['grade'].values, dtype=torch.float32).unsqueeze(1)   # Shape (100,) -> (100, 1)


print("-"*25)
print(X.shape)
print(X.ndim)
print(y.shape)
print(y.ndim)
print("-"*25)

## Split data as Train-Test:
train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split] # 80% of the data for training.
X_test, y_test = X[train_split:], y[train_split:] # 20% of the data for testing.

print(X_train.shape, y_train.shape) # Shape of X_train and y_train variables.
print(X_test.shape, y_test.shape) # Shape of X_test and y_test variables.

## Draw a scatter plot of the data:
plt.scatter(X_train, y_train, c='blue', label='Training Data')
plt.scatter(X_test, y_test, c='red', label='Testing Data')
plt.xlabel('Study Hours')
plt.ylabel('Grade')
plt.title('Study Hours vs Grade')
plt.legend()
plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1) # We need to add one hidden layer to the model. 

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Forward pass.
        return self.linear_layer(x)

## Initialize the model:
torch.manual_seed(42)
model = LinearRegressionModel()
# model = torch.compile(model)


print(list(model.parameters()))
print(list(model.state_dict()))

## Loss function and optimizer:
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)  

## Train model:
epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(100):
    model.train()
    
    # 1. Forward pass
    y_preds = model(X_train)
    
    # 2. Calculate loss
    loss = loss_fn(y_preds, y_train)
    
    # 3. Zero gradients
    optimizer.zero_grad()
    
    # 4. Backpropagation
    loss.backward()
    
    # 5. Update parameters
    optimizer.step()

    model.eval()
    with torch.inference_mode():    
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Test Loss: {test_loss.item()}")
            epoch_count.append(epoch)
            train_loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())

## Plot Loss Curves:
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.show()  
    
print("-"*25)
print(model.state_dict())
print("-"*25)

## Evaluate the model results:
model.eval()
with torch.inference_mode():
    y_preds = model(X_test)

print(X_test)
print(y_preds)

print("\n" + "-"*25)
plt.scatter(X_test, y_test, c='green', label='Actual Values')
plt.scatter(X_test, y_preds, c='red', label='Predicted Values')
plt.legend()
plt.show()  

