# jacrev_finite
`JacrevFinite` is a Python class designed to compute the Jacobian matrix of a given network function using finite differences.

## Installation
PyTorch is needed for implementation.
Download the [jacrev_finite.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file and import the JacrevFinite class:
```bash
from jacrev_finite import JacrevFinite
```
## Usage
### Class definition
`JacrevFinite` class definition can be found in the [jacrev_finite.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/JacrevFinite.py) file
#### Parameters
`JacrevFinite(network=, wrapper=None, dim=None, num_args, delta=1e-5)(input<sub>0</sub>)`
- network: Callable function which takes in input<sub>1</sub> and returns an output
- wrapper: Optional function which takes in input<sub>0</sub> and returns input<sub>1</sub>. If not specified, input<sub>1</sub> is input<sub>0</sub>
- dim: Singleton dimensions over which to append delta tensors to. If not specified, singleton dimension is added at index 0.
- num_args: Specifies over which input<sub>0</sub> the output's derivative should be taken with respect to
- delta: Optional value to change. Default is 1e-5

  ##### Wrapper (optional)
  E.g. if derivative wants to be taken with respect to q_cur, p_cur, q_cur_7, p_cur_7 but the inputs into the network are q_traj and p_traj, a wrapper can be used to convert q_cur, p_cur, q_cur_7, p_cur_7 into q_traj and p_traj after the delta has been added onto one of the 4 initial inputs.

#### Constraints
- Input tensors/lists/tuples/ints/floats must have the same number of dimensions (e.g. (1,16,2) and (2,15,1) etc.)

### Example usage
```bash
def f(x,y):
    return x+y

input = (1,1)
input1 = [2,3]

jacobian = JacrevFinite(network=f, num_args=0)(input, input1)
```
More examples can be found in [Example.py](https://github.com/schrodingerslemur/jacrev_finite/blob/main/Example.py)

### Integration with custom class
Can be found in [LLUF](https://github.com/schrodingerslemur/jacrev_finite/tree/main/LLUF) folder where [LLUF_class](https://github.com/schrodingerslemur/jacrev_finite/tree/main/LLUF/LLUF_class.py) is custom class with network and wrapper methods defined, and [main](https://github.com/schrodingerslemur/jacrev_finite/tree/main/LLUF/main.py) is the code implementation. 

