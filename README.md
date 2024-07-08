# jacrev_finite
`JacrevFinite` is a Python class designed to compute the Jacobian matrix of a given network function using finite differences.

## Installation
PyTorch is needed for implementation.
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
Can be found in Example.py

### Integration with custom class
Can be found in LLUF folder where LLUF/LLUF_class is custom class with network and wrapper methods defined, and LLUF/main is the code implementation. 

