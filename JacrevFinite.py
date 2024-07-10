import torch
from torch import Tensor

class JacrevFinite:
    def __init__(self, *, network, num_args, wrapper=None, dim=None, delta=1e-5):
        """
        Initialize the JacrevFinite object.

        Args:
            network (callable): The network function that takes input2 to output.
            wrapper (callable, optional): A function that takes input1 (with delta added) to input2. Defaults to None.
            output_dim (list or tuple): The dimensions of the network output.
            dim (int, optional): The dimension to append the batch over. Defaults to None.
            num_args (int): The index of the argument in the input list to which the delta is added.

        Raises:
            AssertionError: If num_args is not an int.
            AssertionError: If dim is not an int or None.
        """
        assert isinstance(num_args, int), 'num_args must be int'
        assert isinstance(dim, int) or dim is None, 'dim must be int or None'
    
        self.network = network
        self.wrapper = wrapper
        self.num_args = num_args
        self.delta = delta
        self.dim = dim

    def __call__(self, *args):
        """
        Performs computation.

        Args:
            *args: The input arguments.

        Returns:
            torch.Tensor: The computed Jacobian matrix.
        """
        assert self.num_args < len(args), 'invalid num_args'

        # Converts inputs to a list of tensors
        if len(args) == 1:
            self.inputs = args[0]
            if not isinstance(self.inputs, Tensor):
                self.inputs = torch.tensor(self.inputs)
            self.inputs = self.inputs.unsqueeze(0)
            self.inputs = list(self.inputs)

        else:
            self.inputs = [inputs if isinstance(inputs, Tensor) else torch.tensor(inputs, dtype=torch.float64) for inputs in args]

        first_dim = self.inputs[0].dim()
        for tensor in self.inputs:
            assert tensor.dim() == first_dim, f"Tensor {tensor} has a different number of dimensions: {tensor.dim()} vs {first_dim}"
    
        self.n_inputs = len(args)
        self.output_dim = self.get_outputdim()
        
        # Forward passes
        input1 = self.delta_forward() # changes self.inputs
        input2 = self.wrapper_forward(input1)
        output = self.net_forward(input2)
        jacobian = self.jacobian_forward(output)

        return jacobian
    
    def delta_forward(self):
        """
        Adds delta to the input tensor and appends over specified dimension to create batch tensor.

        Returns:
            list: The list of new inputs with the batch tensor included.
        """
        tensor = self.inputs[self.num_args]
        
        if self.dim is None:
            tensor = tensor.clone().unsqueeze(0)  # Add new singleton dimension
            dim = 0  # The dimension along which to concatenate
        else:
            dim = self.dim  # Use the specified dimension

        assert tensor.size(dim) == 1, 'wrong dimension to add batch to, size must = 1'

        num_rep = tensor.view(-1).size(0) # Number of repetitions (32)
        num_dim = tensor.dim() # (3)

        # Reshape_dim (move dim to last value and multiply by appended size) e.g. (1,16,2) --> (16,2,32) - 1 is moved to last position and multiplied by 32
        reshape_dim = list(tensor.shape)
        reshape_dim.pop(dim)
        reshape_dim.insert(len(reshape_dim), num_rep)

        # Permute_dim (change order of dimensions to move dim to last value) e.g. (0,1,2) --> (1,2,0) - to reshape (16,2,32) back to (32,16,2)
        permute_dim = range(num_dim)
        permute_dim = [num if num<dim else num-1 for num in permute_dim]
        permute_dim[dim] = num_dim-1 

        # Operations to add delta onto every single element: ---------------------
        # Repeat tensor for num_rep rows to obtain square matrix (num_rep rows x num_rep columns)
        repeated_tensor = tensor.view(-1).unsqueeze(0).repeat(num_rep, 1)       # (32,32)
        # Create identity matrix of size (num_rep x num_rep) multiplied by delta
        delta_tensor = torch.eye(num_rep, dtype =tensor.dtype, device=tensor.device)*self.delta     # (32,32)*delta
        # Add the two tensors together
        append_tensor = repeated_tensor + delta_tensor      # (32,32) + (32,32)
      

        # Restructure tensor 
        append_tensor = torch.t(append_tensor)      # Transpose
        append_tensor = append_tensor.reshape(reshape_dim).permute(permute_dim)     # (32,32) --> (16,2,32) --> (32,16,2)


        # Concatenate with original tensor
        batch_tensor = torch.cat((tensor, append_tensor), dim=dim)  # (33,16,2)

        self.batch_size = batch_tensor.size(dim) # (33)

        # Replace original tensor with batch_tensor
        inputs_copy = self.inputs.copy()
        inputs_copy.pop(self.num_args)

        new_inputs = []

        # Make all tensors have the same batch size
        for input_tensor in inputs_copy:
            input_tensor = input_tensor.clone()
            
            if self.dim is None:
                input_tensor = input_tensor.unsqueeze(0)

            repeat_shape = [1] * input_tensor.dim()
            repeat_shape[dim] = self.batch_size
            repeated_tensor = input_tensor.repeat(*repeat_shape)

            new_inputs.append(repeated_tensor)

        new_inputs.insert(self.num_args, batch_tensor)

        return new_inputs
    
    def wrapper_forward(self, input1):
        """
        Apply the wrapper function to input1.

        Args:
            input1 (list): The input list.

        Returns:
            list/tuple/iterable: The output after applying the wrapper.
        """
        if self.wrapper is None:
            input2 = input1
        else:
            # Passed in as *args, not as a list. self.wrapper can be defined as wrapper(self, input0, input1, ...)
            input2 = self.wrapper(*input1)
            
        return input2
    
    def net_forward(self, input2):
        """
        Apply the network function to input2.

        Args:
            input2 (list/tuple/iterable): The input list after wrapping.

        Returns:
            torch.Tensor: The output of the network.
        """
        # Passed in as *args, not as a list. self.network can be defined as network(self, input0, input1, ...)
        output = self.network(*input2)
        return output
    
    def jacobian_forward(self, output):
        """
        Computes the Jacobian matrix.

        Args:
            output (torch.Tensor): The network output.

        Returns:
            torch.Tensor: The computed Jacobian matrix.
        """
        # Compute values for reshape and permutation
        input_delta_shape = list(self.inputs[self.num_args].shape)
        output_shape = self.output_dim
        jacobian_init = input_delta_shape + output_shape

        input_len = len(input_delta_shape)
        output_len = len(output_shape)

        # Initialize the Jacobian with the correct shape
        jacobian_shape = [self.batch_size - 1] + output_shape
        jacobian = torch.empty(jacobian_shape, dtype=output.dtype, device=output.device)

        # Compute the Jacobian using finite differences
        ref = output[0]
        jacobian = (output[1:] - ref) / self.delta

        # Reshape and permute the Jacobian to the desired shape
        jacobian = jacobian.reshape(jacobian_init)
        permute_order = list(range(input_len, input_len + output_len)) + list(range(input_len))
        jacobian = jacobian.permute(*permute_order)

        return jacobian
    
    def get_outputdim(self):
        """
        Gets output dimensions for a single batch.

        Returns:
            list: The output dimensions.
        """
        inputs = self.wrapper_forward(self.inputs)
        output = self.net_forward(inputs)
        output_dim = list(output.shape)

        return output_dim
