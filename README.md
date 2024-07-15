# JacrevFinite
```bash
JacrevFinite(*, network, num_args, wrapper=None, dim=None, override_dim_constraint=False, delta=1e-5)(*args)
```
Computes the jacobian of `network` with respect to the *args at index `num_args` using finite differences as a replacement to `torch.func.jacrev`

### Parameters
- network *(function)* - A Python function which takes one or more arguments and returns one tensor (Note: there is a constraint that the network must only have 1 output)
- num_args *(int)* - Integer which states which arguments to get the Jacobian with respect to.
- wrapper *(function)* - Optional, takes *args and converts it into inputs for networks. Only used in certain cases. i.e. if Jacobian is taken with respect to tensor1 and tensor2, but network can only have an input of type tensor3 and tensor4. Wrapper function can be included to convert tensor1 and tensor2 to tensor3 and tensor4. Make sure that tensor3 and tensor4 are returned as a list: [tensor3, tensor4]. *Default: None*
- dim *(int)* - Integer which states over which singleton dimension to append batch over. *Default: None*
- override_dim_constraint *(bool)* - States whether to override the constraint that all input *args must have the same number of dimensions. *Default: False*
- delta *(float)* - Delta used for finite difference computations. *Default: 1e-5*

### Returns
  Returns the Jacobian of `network` with respect to the *args at `num_args`

## Installation
`PyTorch` is needed for implementation.
Download the [jacrev_finite.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file and import the JacrevFinite class:
```bash
from jacrev_finite import JacrevFinite
```
## Usage
### Class definition
`JacrevFinite` class definition can be found in the [jacrev_finite.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file
#### Parameters
`JacrevFinite(network=, wrapper=None, dim=None, num_args, delta=1e-5)(input0)`
- network: Callable function which takes in input<sub>1</sub> and returns an output
- wrapper: Optional function which takes in input<sub>0</sub> and returns input<sub>1</sub>. If not specified, input<sub>1</sub> is input<sub>0</sub>
- dim: Singleton dimensions over which to append delta tensors to. If not specified, singleton dimension is added at index 0.
- num_args: Specifies over which input<sub>0</sub> the output's derivative should be taken with respect to
- delta: Optional value to change. Default is 1e-5

  ##### Wrapper (optional)
  E.g. if derivative wants to be taken with respect to q_cur, p_cur, q_cur_7, p_cur_7 but the inputs into the network are q_traj and p_traj, a wrapper can be used to convert q_cur, p_cur, q_cur_7, p_cur_7 into q_traj and p_traj after the delta has been added onto one of the 4 initial inputs.

#### Constraints
- Input tensors/lists/tuples/ints/floats must have the same number of dimensions (e.g. (1,16,2) and (2,15,1) etc.)
- Network should only have one output

### Example usage
```bash
def f(x,y):
    return x+y

input1 = (1,1)
input2 = [2,3]

jacobian = JacrevFinite(network=f, num_args=0)(input1, input2)
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

jacobian = JacrevFinite(network=f, wrapper=wrapper, num_args=0)(seq1a, seq1b, seq2a, seq2b)
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

network = neural_net()
jacobian = JacrevFinite(network = network.forward, num_args=0)(input1, input2, input3)
```
Better example found in [LLUF](https://github.com/schrodingerslemur/jacrev_finite/tree/main/LLUF) folder where [LLUF_class](https://github.com/schrodingerslemur/jacrev_finite/tree/main/LLUF/LLUF_class.py) is custom class with network and wrapper methods defined, and [main](https://github.com/schrodingerslemur/jacrev_finite/tree/main/LLUF/main.py) is the code implementation. 

