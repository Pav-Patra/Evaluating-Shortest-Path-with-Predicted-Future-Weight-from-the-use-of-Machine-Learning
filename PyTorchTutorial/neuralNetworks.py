import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Networks - can be constructed using the torch.nm package

# torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor
# nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc
# nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
# autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling ovcer a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
print(net)
print("\n")

# You just have to define the forward function, and the backward function (where gradients are computed) is automatically defined for you using autograd. You can use any of the Tensor operations in the forward function.

# The learnable parameters of a model are returned by net.parameters()

params = list(net.parameters())
print(len(params))
print(params[0].size())
print("\n")

# A random 32x32 input

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
print("\n")

# zero the gradient of all parameters and backdrops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))
print(out, "\n")

# Loss Function
# nn.MSELoss - a simple loss function which computes the mean-squared error between the output and the target. 

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss, "\n")

# follow loss in the backward direction

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])   # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0], "\n")  # ReLU

# Backprop
# To backpropagate the error all we have to do is to loss.backward()
net.zero_grad()  # zeroes the gradient buffers of all parameters
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)
loss.backward()
print("conv1.bias.grad after backward")
print(net.conv1.bias.grad, "\n")


# Update the Weights

#Create your optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()   # Does the update