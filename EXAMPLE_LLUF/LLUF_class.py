import torch

class LLUF_jacrev():
    """
    Class to perform Jacobian computations using a given network and wrapper functions.
    Necessary methods for JacrevFinite implementation:
    - self.wrapper: Wrapper function to convert input1(s) to input2(s) (combine current and trajectory tensors).
    - self.network: Forward pass through the network taking in input2(s)

    Args:
        networks (list): A list containing the network components. It should contain:
                         - prep_net: A network for preparing the data.
                         - LLUF_net: A network for updating the data.
                         - l_init: Initial values for the network.
    """

    def __init__(self, networks):
        self.prep_net = networks[0]
        self.LLUF_net = networks[1]
        self.l_init = networks[2]
        self.prepare_q = self.prep_net.prepare_q_feature_input
        self.prepare_p = self.prep_net.prepare_p_feature_input
        self.pforward = self.LLUF_net.LLUF_update_p1st
        self.qforward = self.LLUF_net.LLUF_update_q
    
    def q_network(self, q_cur, p_cur, q_traj, p_traj):
        """
        Forward pass through the network.

        Args:
            q_cur, p_cur, q_traj, p_traj

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        # Squeeze the batch dimension from current position tensors
        q_cur = q_cur.squeeze(0)
        p_cur = p_cur.squeeze(0)

        # Initialize empty tensors for input features
        q_input = torch.empty((8, 0, 16, 12), dtype=torch.float64, device='cuda')
        p_input = torch.empty((8, 0, 16, 12), dtype=torch.float64, device='cuda')

        batch_size = q_traj.size(1)

        # Process each batch
        for i in range(batch_size):
            q_traj_batch = q_traj[:, i, :, :].unsqueeze(1)
            p_traj_batch = p_traj[:, i, :, :].unsqueeze(1)

            q_traj_batch_list = list(q_traj_batch)
            p_traj_batch_list = list(p_traj_batch)

            q_input_list, p_input_list = [], []

            # Prepare q and p features
            for q, p in zip(q_traj_batch_list, p_traj_batch_list):
                q_input_list.append(self.prepare_q(q, self.l_init))
                p_input_list.append(self.prepare_p(q, p, self.l_init))

            q_input_tensor = torch.cat(q_input_list, dim=0).unsqueeze(1)
            p_input_tensor = torch.cat(p_input_list, dim=0).unsqueeze(1)

            q_input = torch.cat((q_input, q_input_tensor), dim=1)
            p_input = torch.cat((p_input, p_input_tensor), dim=1)
        
        q_input_list = list(q_input)
        p_input_list = list(p_input)

        # Perform the forward pass
        h = self.qforward(q_input_list, p_input_list, q_cur)
        return h
    
    def p_network(self, q_cur, p_cur, q_traj, p_traj):
        """
        Forward pass through the network.

        Args:
            q_cur, p_cur, q_traj, p_traj

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        # Squeeze the batch dimension from current position tensors
        q_cur = q_cur.squeeze(0)
        p_cur = p_cur.squeeze(0)

        # Initialize empty tensors for input features
        q_input = torch.empty((8, 0, 16, 12), dtype=torch.float64, device='cuda')
        p_input = torch.empty((8, 0, 16, 12), dtype=torch.float64, device='cuda')

        batch_size = q_traj.size(1)

        # Process each batch
        for i in range(batch_size):
            q_traj_batch = q_traj[:, i, :, :].unsqueeze(1)
            p_traj_batch = p_traj[:, i, :, :].unsqueeze(1)

            q_traj_batch_list = list(q_traj_batch)
            p_traj_batch_list = list(p_traj_batch)

            q_input_list, p_input_list = [], []

            # Prepare q and p features
            for q, p in zip(q_traj_batch_list, p_traj_batch_list):
                q_input_list.append(self.prepare_q(q, self.l_init))
                p_input_list.append(self.prepare_p(q, p, self.l_init))

            q_input_tensor = torch.cat(q_input_list, dim=0).unsqueeze(1)
            p_input_tensor = torch.cat(p_input_list, dim=0).unsqueeze(1)

            q_input = torch.cat((q_input, q_input_tensor), dim=1)
            p_input = torch.cat((p_input, p_input_tensor), dim=1)
        
        q_input_list = list(q_input)
        p_input_list = list(p_input)

        # Perform the forward pass
        h = self.pforward(q_input_list, p_input_list, q_cur)
        return h
    
    def wrapper(self, q_cur, p_cur, q_traj_7, p_traj_7):
        """
        Wrapper function to combine current and trajectory tensors.

        Args:
            q_cur, p_cur, q_traj_7, p_traj_7

        Returns:
            list: A list containing [q_cur, p_cur, q_traj, p_traj].
        """
        q_traj = torch.cat([q_traj_7, q_cur], dim=0)
        p_traj = torch.cat([p_traj_7, p_cur], dim=0)
        
        input2 = [q_cur, p_cur, q_traj, p_traj]
        return input2
    
    def p_forward(self, q_cur, p_cur, q_traj_7, p_traj_7):
        """
        Full forward pass through the network including data preparation. For jacrev implementation.

        Args:
            q_cur (torch.Tensor): Current q positions.
            p_cur (torch.Tensor): Current p positions.
            q_traj_7 (torch.Tensor): Trajectory q positions.
            p_traj_7 (torch.Tensor): Trajectory p positions.

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        q_traj = torch.cat([q_traj_7, q_cur], dim=0)
        p_traj = torch.cat([p_traj_7, p_cur], dim=0)
 
        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        q_input_list = []
        p_input_list = []

        # Prepare q and p features
        for q, p in zip(q_traj_list, p_traj_list):
            q_input_list.append(self.prepare_q(q, self.l_init))
            p_input_list.append(self.prepare_p(q, p, self.l_init))

        # Perform the forward pass
        h_p1st = self.LLUF_net.LLUF_update_p1st(q_input_list, p_input_list, q_cur)
        return h_p1st
    
    def q_forward(self, q_cur, p_cur, q_traj_7, p_traj_7):
        """
        Full forward pass through the network including data preparation. For jacrev implementation.

        Args:
            q_cur (torch.Tensor): Current q positions.
            p_cur (torch.Tensor): Current p positions.
            q_traj_7 (torch.Tensor): Trajectory q positions.
            p_traj_7 (torch.Tensor): Trajectory p positions.

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        q_traj = torch.cat([q_traj_7, q_cur], dim=0)
        p_traj = torch.cat([p_traj_7, p_cur], dim=0)
 
        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        q_input_list = []
        p_input_list = []

        # Prepare q and p features
        for q, p in zip(q_traj_list, p_traj_list):
            q_input_list.append(self.prepare_q(q, self.l_init))
            p_input_list.append(self.prepare_p(q, p, self.l_init))

        # Perform the forward pass
        h_p1st = self.LLUF_net.LLUF_update_q(q_input_list, p_input_list, q_cur)
        return h_p1st
    
    def preprocess(self, q_traj, p_traj):
        """
        Preprocess the trajectory tensors.

        Args:
            q_traj (torch.Tensor): Trajectory q positions.
            p_traj (torch.Tensor): Trajectory p positions.

        Returns:
            tuple: A tuple containing (q_cur, p_cur, q_traj_7, p_traj_7).
        """
        q_traj_7, q_cur = q_traj[:7], q_traj[7].unsqueeze(0)
        p_traj_7, p_cur = p_traj[:7], p_traj[7].unsqueeze(0)
        
        return q_cur, p_cur, q_traj_7, p_traj_7
