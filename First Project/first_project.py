# First project with Python for Deep Learning Courses.
# We will be using the Pandas, Numpy, Matplotlib and Torch libraries to analyze the data.

## Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

## Data:
data = pd.read_csv('study_hours_grades.csv')

## Exploratory Data Analysis:
print(data.head())
print(data.info())
print(data.describe().T)

## Convert the numpy array to a tensor:
type(torch.tensor(data['study_hours'].values)) # Type is Tensor.
type(torch.tensor(data['study_hours'].values)) # Type is Tensor.

## Describe X and y variables:
y = torch.tensor(data['grade'].values)
X = torch.tensor(data['study_hours'].values)

print(X.shape, y.shape) # Shape of X and y variables.

## Test-Train split:
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

## Build a simple artifical neural network:
class SimpleLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True) # Randomly initialize the weights.
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True) # Randomly initialize the bias.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias # Forward pass.

## Initialize the model:
torch.manual_seed(42) # Set the random seed for reproducibility.
model = SimpleLinearRegressionModel()


## State_Dict and Parameters:
print(list(model.parameters()))
print(list(model.state_dict()))

## Make predictions:
with torch.inference_mode():
    y_preds = model(X_test)
print(y_preds)
print(y_test)

## Visualize the model:
plt.scatter(X_train, y_train, c='blue', label='Training Data')
plt.scatter(X_test, y_test, c='green', label='Testing Data')
plt.scatter(X_test, y_preds, c='red', label='Predicted Data')
plt.legend()
plt.show() # We didn't train the model yet, so the predictions are not good.

## Loss Function and Optimizer:
loss_fn = nn.MSELoss() # loss_fn = nn.L1loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001) 

## Train the model:
epoch_count = []
train_loss_values = []
test_loss_values = []

torch.manual_seed(42)
for epoch in range(200):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad() # This three steps are being backpropagation.
    loss.backward() #
    optimizer.step() # 

    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy()) # Matplotlib Library doesn't support torch tensor so we have to convert it to numpy array.
            test_loss_values.append(test_loss.detach().numpy()) # Same as above.
            print(f"Epoch: {epoch}, Loss: {loss.detach().numpy()}, Test Loss: {test_loss.detach().numpy()}")

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


