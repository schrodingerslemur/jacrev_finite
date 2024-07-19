# JacrevFinite
```bash
JacrevFinite(*, function, num_args, wrapper=None, dim=None, override_dim_constraint=False, delta=1e-5, method='plus')(*args)
```
**`JacrevFinite`** computes the Jacobian of a given **`function`** with respect to the arguments at the specified **`num_args`** index using finite differences, providing an alternative to **`torch.func.jacrev`**.

### Parameters
- **function** *(function)*: A Python function that takes one or more arguments and returns a single tensor. Note: the function must have only one output.
- **num_args** *(int)*: Index of the arguments to compute the Jacobian with respect to.
- **wrapper** *(function, optional)*: A function to convert *args into inputs for the main function, used when the main function cannot directly accept *args. The wrapper should return a list of transformed inputs. *Default: None*
- **dim** *(int, optional)*: Specifies the dimension to append batches over. If **`None`**, a singleton dimension at dimension 0 is added. *Default: None*
- **override_dim_constraint** *(bool, optional)*: Allows overriding the constraint that all input arguments must have the same number of dimensions. *Default: False*
- **delta** *(float, optionol)*: Step size used for finite difference computations. The most stable delta values are between 1e-4 and 1e-5. *Default: 1e-5*
- **method** *(str, optional)*: Either 'plus' or 'minus'. Specifies whether delta should be added or subtracted for finite difference computations. Both methods should yield similar results but can be interchanged if accuracy is sub-par. *Default: 'plus'*
  
### Returns
Returns the Jacobian of **`function`** with respect to the arguments at index **`num_args`**.
  
### How it works  
The function takes the input at index num_args (e.g., a tensor of size [1,16,2]) and creates multiple batches where delta is added to each element. The batches are passed through the function, and the Jacobian is calculated using the finite difference method 
(
𝑓
(
𝑥
+
ℎ
)
−
𝑓
(
𝑥
)
)
/
𝛿
.

### Wrapper (optional)
If the Jacobian is needed with respect to specific inputs but the function requires different inputs, a wrapper can be used. For example, if the derivative is taken with respect to **`q_cur`**, **`p_cur`**, **`q_cur_7`**, **`p_cur_7`** but the function inputs are **`q_traj`** and **`p_traj`**, the wrapper can convert **`q_cur`**, **`p_cur`**, **`q_cur_7`**, **`p_cur_7`** to **`q_traj`** and **`p_traj`**.
#### Constraints
- Input arguments (tensors, lists, tuples, ints, floats) must have the same number of dimensions.
- The function should have only one output.

## Installation
**`PyTorch`** is required.
Download the [**JacrevFinite.py**](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file and import the JacrevFinite class:
```bash
from JacrevFinite import JacrevFinite
```
Run the following code after downloading [**JacrevFinite.py**](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file. Ensure the output prints five `True` statements.

```bash
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
```

## Examples
### Example usage
```bash
def f(x,y):
    return x+y

input1 = (1,1)
input2 = [2,3]

jacobian = JacrevFinite(function=f, num_args=0)(input1, input2)
```

### Example usage with wrapper
```bash
def f(x,y):
    return x+y

seq1a = [1,2,3,4]
seq1b = [5,6,7,8]

seq2a = [-1,-2,-3,-4]
seq2b = [-5,-6,-7,-8]

def wrapper(seq1a, seq1b, seq2a, seq2b):
    seq1 = seq1a + seq1b # Concatenate
    seq2 = seq2a + seq2b
    return [seq1, seq2]

jacobian = JacrevFinite(function=f, wrapper=wrapper, num_args=0)(seq1a, seq1b, seq2a, seq2b)
```
More examples can be found in [testcases.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/testcases.py)

### Integration with custom class
```bash
class update():
...
    def forward(self, x, y, z):
        ...
        return output

input1 = torch.randn(2,3)
input2 = torch.randn(2,3)
input3 = torch.randn(2,3)

function = update()
jacobian = JacrevFinite(function = function.forward, num_args=0)(input1, input2, input3)
```
A better examples is found in the [EXAMPLE_LLUF](https://github.com/schrodingerslemur/jacrev_finite/tree/main/EXAMPLE_LLUF) folder where [LLUF_class](https://github.com/schrodingerslemur/jacrev_finite/tree/main/EXAMPLE_LLUF/LLUF_class.py) defines custom class with function and wrapper methods, and [main](https://github.com/schrodingerslemur/jacrev_finite/tree/main/EXAMPLE_LLUF/main.py) contains the code implementation.
