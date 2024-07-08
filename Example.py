import torch
from torch.func import jacrev
from jacrev_finite import JacrevFinite

# SIMPLE IMPLEMENTATION FROM DEFINED FUNCTION:
def two(x):
    return 2*x

input = torch.tensor((5,2), dtype = torch.float64)
output = two(input)

j = jacrev(two, argnums=0)(input)
print(j)

t = JacrevFinite(network=two, num_args=0)(input)
print(t)

# IMPLEMENTATION FROM DEFINED CLASS:
# q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label) # Has to be implemented in main file of SJ's code

# networks = [train.prepare_data_obj, train.mlvv, l_init]
# lluf = LLUF_jacrev(networks) 
# q_cur, p_cur, q_traj_7, p_traj_7 = lluf.preprocess(q_traj, p_traj)

# print('-------------------------------------q---------------------------------------')
# jacobian = JacrevFinite(network=lluf.network,dim=1,wrapper=lluf.wrapper, num_args=0)(q_cur, p_cur, q_traj_7, p_traj_7)
# jacobian = jacobian.sum(dim=(0,1,2,3,4)) # jacobian = [outputdim, inputdim]

# jacobian_actual = jacrev(lluf.full_forward, argnums=0)(q_cur, p_cur, q_traj_7, p_traj_7)
# jacobian_actual = jacobian_actual.sum(dim=(0,1,2,3,4))

# print('------------------------------------p----------------------------------------')
# jacobian = JacrevFinite(network=lluf.network,dim=1,wrapper=lluf.wrapper, num_args=1)(q_cur, p_cur, q_traj_7, p_traj_7)
# jacobian = jacobian.sum(dim=(0,1,2,3,4)) # jacobian = [outputdim, inputdim]

# jacobian_actual = jacrev(lluf.full_forward, argnums=1)(q_cur, p_cur, q_traj_7, p_traj_7)
# jacobian_actual = jacobian_actual.sum(dim=(0,1,2,3,4))