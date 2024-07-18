# JacrevFinite
```bash
JacrevFinite(*, function, num_args, wrapper=None, dim=None, override_dim_constraint=False, delta=1e-5)(*args)
```
Computes the jacobian of `function` with respect to the *args at index `num_args` using finite differences as a replacement to `torch.func.jacrev`

### Parameters
- function *(function)* - A Python function which takes one or more arguments and returns one tensor (Note: there is a constraint that the function must only have 1 output)
- num_args *(int)* - Integer which states which arguments to get the Jacobian with respect to.
- wrapper *(function)* - Optional, takes *args and converts it into inputs for functions. Only used in certain cases. i.e. if Jacobian is taken with respect to tensor1 and tensor2, but function can only have an input of type tensor3 and tensor4. Wrapper function can be included to convert tensor1 and tensor2 to tensor3 and tensor4. Make sure that tensor3 and tensor4 are returned as a list: [tensor3, tensor4]. *Default: None*
- dim *(int)* - Integer which states over which singleton dimension to append batch over. If None, a singleton dimension at dimension 0 is added. *Default: None*
- override_dim_constraint *(bool)* - States whether to override the constraint that all input *args must have the same number of dimensions. *Default: False*
- delta *(float)* - Delta used for finite difference computations. *Default: 1e-5*
### Returns
  Returns the Jacobian of `function` with respect to the *args at index `num_args`
  
##### Wrapper (optional)
  E.g. if derivative wants to be taken with respect to q_cur, p_cur, q_cur_7, p_cur_7 but the inputs into the function are q_traj and p_traj, a wrapper can be used to convert q_cur, p_cur, q_cur_7, p_cur_7 into q_traj and p_traj after the delta has been added onto one of the 4 initial inputs.
#### Constraints
- Input tensors/lists/tuples/ints/floats must have the same number of dimensions (e.g. (1,16,2) and (2,15,1) etc.)
- Function should only have one output

## Installation
`PyTorch` is needed for implementation.
Download the [JacrevFinite.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file and import the JacrevFinite class:
```bash
from JacrevFinite import JacrevFinite
```
Copy and run the following code after downloading [JacrevFinite.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file and ensure the output returns `True`. 
*The following code can also be found in [testcases.py](https://github.com/schrodingerslemur/jacrev_finite/edit/main/testcases.py)*
```bash
import torch
from torch.func import jacrev
from jacrev_finite import JacrevFinite

# Assert values are similiar to torch.func.jacrev
input1 = torch.randn((100,100), dtype=torch.float64)
input2 = torch.randn((100,100), dtype=torch.float64)

def function(x,y):
    return x*y

def assertTensorEqual(a, b, abs_tol=1e-9, mean_tol=1e-14):
    mean = a.sub(b).abs().mean().item()
    max = a.sub(b).abs().max().item()
    print(f"Error with actual jacrev // mean error: {mean}, max error: {max}")
    return (max<abs_tol and mean<mean_tol)

jacobian_auto = jacrev(func=function, argnums=0)(input1, input2)
jacobian_finite = JacrevFinite(function=function, num_args=0)(input1, input2)
    
print(assertTensorEqual(jacobian_auto, jacobian_finite))
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
More examples can be found in [Example.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/Example.py)

### Integration with custom class
```bash
class neural_net():
...
    def forward(self, x, y, z):
        ...
        return output

input1 = torch.randn(2,3)
input2 = torch.randn(2,3)
input3 = torch.randn(2,3)

function = neural_net()
jacobian = JacrevFinite(function = function.forward, num_args=0)(input1, input2, input3)
```
Better example found in [LLUF](https://github.com/schrodingerslemur/jacrev_finite/tree/main/EXAMPLE_LLUF) folder where [LLUF_class](https://github.com/schrodingerslemur/jacrev_finite/tree/main/EXAMPLE_LLUF/LLUF_class.py) is custom class with function and wrapper methods defined, and [main](https://github.com/schrodingerslemur/jacrev_finite/tree/main/EXAMPLE_LLUF/main.py) is the code implementation. 

