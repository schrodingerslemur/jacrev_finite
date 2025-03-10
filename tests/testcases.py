import torch
import torch.nn as nn
from torch.func import jacrev
from JacrevFinite import JacrevFinite

def function(x,y):
    return x*y

class Network(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        fc1 = nn.Linear(n_input, n_hidden)
        fc2 = nn.Linear(n_hidden, n_hidden)
        fc3 = nn.Linear(n_hidden, n_hidden)
        fc4 = nn.Linear(n_hidden, n_hidden)
        fc5 = nn.Linear(n_hidden, n_output)

        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4,fc5])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = torch.relu(x)
        return x 
    
def assertTensorEqual(a, b, abs_tol=1e-10, mean_tol=1e-10):
    mean = a.sub(b).abs().mean().item()
    max = a.sub(b).abs().max().item()
    isEqual = (max<abs_tol and mean<mean_tol)
    print(isEqual)
    print(f"Error:\nmean error: {mean}, max error: {max}")
    pass

# Assert values are similiar to torch.func.jacrev
input1 = torch.randn((100,100), dtype=torch.float64)
input2 = torch.randn((100,100), dtype=torch.float64)

jacobian_auto = jacrev(func=function, argnums=0)(input1, input2)
jacobian_finite = JacrevFinite(function=function, num_args=0)(input1, input2)
    
assertTensorEqual(jacobian_auto, jacobian_finite)

# Assert values can be appended over different dim
input3 = torch.randn((64,1,64), dtype=torch.float64)
input4 = torch.randn((64,1,64), dtype=torch.float64)
input5 = torch.randn

jacobian_auto1 = jacrev(func=function, argnums=0)(input3, input4)
jacobian_finite1 = JacrevFinite(function=function, num_args=0)(input3,input4)
jacobian_finite2 = JacrevFinite(function=function, num_args=0, dim=1)(input3, input4)

assertTensorEqual(jacobian_finite1, jacobian_finite2)
assertTensorEqual(jacobian_auto1, jacobian_finite1)

# Compare values for network forward passes
net = Network(5,5,128).double()

input6 = torch.randn((20,5), dtype=torch.float64)

jacobian_auto2 = jacrev(func=net, argnums=0)(input6)
jacobian_finite3 = JacrevFinite(function=net, num_args=0)(input6)

assertTensorEqual(jacobian_auto2, jacobian_finite3)

# Bigger dimensions for network forward passes
net = Network(2,2,256).double()

input7 = torch.randn((8,1,16,2), dtype=torch.float64)

jacobian_auto3 = jacrev(func=net, argnums=0)(input7)
jacobian_finite4 = JacrevFinite(function=net, num_args=0, dim=1)(input7)

assertTensorEqual(jacobian_auto3, jacobian_finite4)
