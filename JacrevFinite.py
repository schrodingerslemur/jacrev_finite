import torch
from torch import Tensor

class JacrevFinite:
    def __init__(self, *, function, num_args, wrapper=None, dim=None, delta=1e-5, override_dim_constraint=False, method='plus'):
        """
        Initialize JacrevFinite object.

        Args:
            function (callable): Function that takes one or more arguments and returns a single tensor.
            num_args (int): Index of the arguments to compute the Jacobian with respect to
            wrapper (callable, optional): Function to convert *args into inputs for main function, used when main function cannot directly accept *args. 
                Wrapper should return list of transformed inputs. Default: None
            dim (int, optional): Specifies the dimension to append batches over. If None, a singleton dimension at dimension 0 is added.
                Must be a singleton dimension.
            delta (float, optional): Step size used for finite difference computations. Most stable at 1e-5 or 1e-4. Default: 1e-5
            override_dim_constraint (bool, optional): Overrides constraint that input arguments must have same number of dimensions. Default: False
            method (str, optional): Either 'plus' or 'minus'. Specifies whether delta should be added or subtracted for finite difference computations. 
                Both methods should yield similar results but can be interchanged if accuracy is sub-par. Default: 'plus'         

        Constraints:
            Inputs must have the same number of dimensions (.dim() must be equal)
            Function must only have one output

        Raises:
            AssertionError: If num_args is not an int.
            AssertionError: If dim is not an int or None.
            AssertionError: if override_dim_constraint is not bool.
            AssertionError: If method is not 'plus' or 'minus'
        """
        assert isinstance(num_args, int), 'num_args must be int'
        assert isinstance(dim, int) or dim is None, 'dim must be int or None'
        assert isinstance(override_dim_constraint, bool), 'override_dim_constraint must be bool'
        assert method in ['plus', 'minus'], 'method must be \'plus\' or \'minus\''
    
        self.function = function
        self.wrapper = wrapper
        self.num_args = num_args
        self.delta = delta
        self.dim = dim
        self.override = override_dim_constraint
        self.method = method

    def __call__(self, *args):
        """
        Performs computation.

        Args:
            *args: Input arguments.

        Returns:
            Tensor: Jacobian matrix.
        """
        assert self.num_args < len(args), 'invalid num_args'

        # Converts inputs to a list of tensors
        if len(args) == 1:
            self.inputs = args[0]
            if not isinstance(self.inputs, Tensor):
                self.inputs = torch.tensor(self.inputs)
            self.inputs = self.inputs.unsqueeze(0).to(torch.float64)
            self.inputs = list(self.inputs)

        else:
            self.inputs = [inputs if isinstance(inputs, Tensor) else \
                           torch.tensor(inputs, dtype=torch.float64) for inputs in args]
        
        # Checks that all the tensors have the same number of dimensions
        if self.override is False:
            first_dim = self.inputs[0].dim()
            for tensor in self.inputs:
                assert tensor.dim() == first_dim, f"Tensor {tensor} has a different number of dimensions: \
                    {tensor.dim()} vs {first_dim}"
    
        self.output_dim = self.get_outputdim()
        
        # Forward passes
        input1 = self.delta_forward() # changes self.inputs
        input2 = self.wrapper_forward(input1)
        output = self.func_forward(input2)
        jacobian = self.jacobian_forward(output)

        return jacobian
    
    def delta_forward(self):
        """
        Creates batch tensor by repeating input tensors and adding delta to 1 element per repeated tensor.

        Returns:
            list: List of new inputs with the batch tensor included.
        """
        # Specifies which tensor to append delta over
        tensor = self.inputs[self.num_args]
        
        if self.dim is None:
            tensor = tensor.clone().unsqueeze(0)  # Add new singleton dimension
            dim = 0  # The dimension along which to concatenate
        else:
            dim = self.dim  # Use the specified dimension

        assert tensor.size(dim) == 1, 'wrong dimension to append batch over, size must = 1'

        num_rep = tensor.view(-1).size(0) # Number of repetitions 
        num_dim = tensor.dim() # Number of dimensions in tensor

        # Repeat_dim (num_rep times over dim)
        repeat_dim = torch.ones(num_dim, dtype=int).tolist()
        repeat_dim[dim] = num_rep

        # Reshape_dim (move dim to last value and multiply by appended size) 
        reshape_dim = list(tensor.shape)
        reshape_dim.pop(dim)
        reshape_dim.insert(len(reshape_dim), num_rep)

        # Permute_dim (change order of dimensions to move dim to last value) 
        permute_dim = range(num_dim)
        permute_dim = [num if num<dim else num-1 for num in permute_dim]
        permute_dim[dim] = num_dim-1 

        # Operations to add delta onto every single element: ---------------------

        # Repeat tensor num_rep times over dim
        repeated_tensor = tensor.repeat(repeat_dim)     

        # Create identity matrix of size (num_rep x num_rep) multiplied by delta then reshape to fit repeated_tensor
        delta_tensor = torch.eye(num_rep, dtype =tensor.dtype, device=tensor.device)*self.delta   
        delta_tensor = delta_tensor.reshape(reshape_dim).permute(permute_dim)

        # Add or minus the tensors together
        if self.method == 'plus':
            append_tensor = repeated_tensor + delta_tensor
        else:
            append_tensor = repeated_tensor - delta_tensor

        # Concatenate with original tensor
        batch_tensor = torch.cat((tensor, append_tensor), dim=dim)  
        self.batch_size = batch_tensor.size(dim)

        # Replace inputs with batch_tensor and ensure all tensors have same batch size: --------------------------
        inputs_copy = self.inputs.copy()
        inputs_copy.pop(self.num_args)

        new_inputs = []

        # Repeating other tensors to ensure same batch size
        for input_tensor in inputs_copy:
            input_tensor = input_tensor.clone()
            
            if self.dim is None:
                input_tensor = input_tensor.unsqueeze(0)

            repeat_shape = [1] * input_tensor.dim()
            repeat_shape[dim] = self.batch_size
            repeated_tensor = input_tensor.repeat(*repeat_shape)

            new_inputs.append(repeated_tensor)

        new_inputs.insert(self.num_args, batch_tensor)

        self.new_dim = dim
        return new_inputs
    
    def wrapper_forward(self, input1):
        """
        Apply the wrapper function to input1.

        Args:
            input1 (list): Input list from delta_forward.

        Returns:
            list/tuple/iterable: Output after applying the wrapper.
        """
        if self.wrapper is None:
            input2 = input1
        else:
            input2 = self.wrapper(*input1)
            
        return input2
    
    def func_forward(self, input2):
        """
        Apply the function to input2.

        Args:
            input2 (list/tuple/iterable): Input list from wrapper_forward.

        Returns:
            Tensor: Output of the function.
        """
        output = self.function(*input2)
        return output
    
    def jacobian_forward(self, output):
        """
        Computes the Jacobian matrix.

        Args:
            output (Tensor): Output from func_forward.

        Returns:
            Tensor: Computed Jacobian matrix.
        """
        # Compute values for reshape and permutation
        input_delta_shape = list(self.inputs[self.num_args].shape)
        output_shape = self.output_dim
        jacobian_init = input_delta_shape + output_shape

        input_len = len(input_delta_shape)
        output_len = len(output_shape)

        # Determine over which dimension to do finite difference (subtract and divide delta)
        batch_output_shape = list(output.shape)
        dim = batch_output_shape.index(self.batch_size)

        # Finite difference to obtain Jacobian
        ref = output.select(dim,0)
        output_transposed = output.transpose(0, dim)
        jacobian = (output_transposed[1:] - ref)/self.delta

        # Reshape and permute the Jacobian to the desired shape
        jacobian = jacobian.reshape(jacobian_init)
        permute_order = list(range(input_len, input_len + output_len)) + list(range(input_len))
        jacobian = jacobian.permute(*permute_order)
        
        # For negative delta instance
        if self.method == 'minus':
            jacobian = torch.neg(jacobian) # Changed
        return jacobian
    
    def get_outputdim(self):
        """
        Gets output dimensions for a single batch.
        Used to determine dimensions of Jacobian matrix

        Returns:
            list: The output dimensions.
        """
        inputs = self.wrapper_forward(self.inputs)
        output = self.func_forward(inputs)
        output_dim = list(output.shape)

        return output_dim
