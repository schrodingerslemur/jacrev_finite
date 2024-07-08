# jacrev_finite
`JacrevFinite` is a Python class designed to compute the Jacobian matrix of a given network function using finite differences.

## Installation
Ensure you have Python and PyTorch installed. You can install PyTorch via pip:
```bash
pip install torch
```

Download the jacrev_finite.py file and import the JacrevFinite class:
```bash
from jacrev_finite import JacrevFinite
```
## Usage
### Class definition
JacrevFinite class definition can be found in the jacrev_finite.py file
#### Parameters
- network: A callable (typically a PyTorch model) that takes input tensors and produces an output tensor. This is the function whose Jacobian you want to compute.
- wrapper: An optional callable that processes the input tensors before passing them to the network. This can be used to transform or preprocess the inputs. If not provided, the inputs are passed directly to the network.
- dim: An optional integer specifying the dimension along which to append the batch. If None, a new dimension is added at the beginning.
- num_args: An integer specifying the index of the input argument to which the delta is added. This argument will be perturbed to compute the finite differences.

#### Constraints
- Only tensors can be passed through the model
- Input tensors must have the same number of dimensions (e.g. (1,16,2) and (2,15,1) etc.)

### Example usage
Here's an example of how to use the 'JacrevFinite' class, which can also be found in Example.py:
```bash
import torch
from torch.func import jacrev
from jacrev_finite import JacrevFinite

def two(x):
    if isinstance(x, list):
        x = torch.stack(x, dim=0)
    return 2 * x

input = torch.tensor((5, 2), dtype=torch.float64)

# Using torch.func.jacrev
j = jacrev(two, argnums=0)(input)
print(j)

# Using JacrevFinite
t = JacrevFinite(network=two, num_args=0)(input)
print(t)
```

### Integration with custom class
You can also integrate 'JacrevFinite' with your custom class. For instance, if you have a custom class 'LLUF_class', like in LLUF/LLUF_class.py, you can use it as follows:
```bash
# Assuming LLUF_jacrev class is defined
lluf = LLUF_jacrev([prep_net, LLUF_net, l_init])

# Prepare the input tensors
q_cur, p_cur, q_traj_7, p_traj_7 = lluf.preprocess(q_traj, p_traj)

# Compute the Jacobian using JacrevFinite
jacobian = JacrevFinite(network=lluf.network, dim=1, wrapper=lluf.wrapper, num_args=0)(q_cur, p_cur, q_traj_7, p_traj_7)
jacobian = jacobian.sum(dim=(0,1,2,3,4))  # Jacobian = [outputdim, inputdim]

print(jacobian)
```

