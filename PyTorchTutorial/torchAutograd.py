import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# autograd is PyTorch's automatic differentiation engione that powers neural network training. 
# Neural Networks (NN) - a collection of nested functions that are executed on some input data.

# single training set
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward()  # backward pass

# Next, load an optimizer (SGD) with a learning rate of 0.01 and momentum 0.9. We register all the parameters if the model in the optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# finally call .step() to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in .grad
optim.step()

# Differentiation in Autograd
a = torch.tensor([2., 3.], requires_grad=True)    # requires_grad - signals autograd that every operation on them should be tracked
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

# Vector calculus using autograd

# Exclusion from the DAG
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does 'a' require gradients? : {a.requires_grad}")
print(f"Does 'b' require gradients? : {b.requires_grad}")

# In a NN, parameters thaty don't compte gradients are usually called frozen parameters

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# let's say we want to finetun the model on a new dataset with 10 labels, in resnet the classifier is the last linear layer model.fc
model.fc = nn.Linear(512, 10)

# Now all the parameters in the model, except the parameters of model.fc, are frozen. The only parameters that compute gradients are the weights and bias of model.fc

# Optimize only the classifier
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)