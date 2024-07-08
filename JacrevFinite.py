import torch

class JacrevFinite:
    def __init__(self, *, network, wrapper=None, dim=None, num_args):
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
        self.delta = 1e-5
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
        
        self.inputs = list(args)
        self.n_inputs = len(args)
        self.output_dim = self.get_outputdim()
        
        input1 = self.delta_forward()
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
        flat_tensor = tensor.view(-1)

        if self.dim is None:
            batch_tensor = tensor.clone().unsqueeze(0)  # Add new singleton dimension
            dim = 0  # The dimension along which to concatenate
        else:
            batch_tensor = tensor.clone()
            dim = self.dim  # Use the specified dimension

        assert batch_tensor.size(dim) == 1, 'wrong dimension to add batch to, size must = 1'

        for i in range(flat_tensor.size(0)):
            delta_tensor = tensor.clone()
            delta_tensor.view(-1)[i] += self.delta
            if self.dim is None:
                delta_tensor = delta_tensor.unsqueeze(0)
            batch_tensor = torch.cat((batch_tensor, delta_tensor), dim=dim)

        self.batch_size = batch_tensor.size(dim)

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
            list: The output list after applying the wrapper.
        """
        if self.wrapper is None:
            input2 = input1
        else:
            input2 = self.wrapper(input1)
        return input2
    
    def net_forward(self, input2):
        """
        Apply the network function to input2.

        Args:
            input2 (list): The input list after wrapping.

        Returns:
            torch.Tensor: The output of the network.
        """
        if isinstance(input2, list):
            output = self.network(*input2)
        else:
            output = self.network(input2)
        return output
    
    def jacobian_forward(self, output):
        """
        Computes the Jacobian matrix.

        Args:
            output (torch.Tensor): The network output.

        Returns:
            torch.Tensor: The computed Jacobian matrix.
        """
        input_delta_shape = list(self.inputs[self.num_args].shape)
        output_shape = self.output_dim
        jacobian_init = input_delta_shape + output_shape

        input_len = len(input_delta_shape)
        output_len = len(output_shape)

        # Initialize the Jacobian with the correct shape
        jacobian_shape = [self.batch_size - 1] + output_shape
        jacobian = torch.empty(jacobian_shape, dtype=output.dtype, device=output.device)

        # Compute the Jacobian using finite differences
        for i in range(self.batch_size-1):
            jacobian[i] = (output[i+1] - output[0]) / self.delta

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
        # if isinstance(output, list):
        #     if self.dim is None:
        #         output = torch.stack(output, dim=0)
        #     else:
        #         output = torch.stack(output, dim=self.dim)
        output_dim = list(output.shape)
        return output_dim

