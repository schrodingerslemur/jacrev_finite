import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append('../')
from LLUF_june4.utils.mydevice import mydevice
from LLUF_june4.ML.LLUF.hamiltonian_derivative import hamiltonian_derivative
from torch.autograd import grad
from LLUF_june4.utils.pbc import pbc
from LLUF_june4.utils.utils import assert_nan

class LLUF_MD(nn.Module):

    def __init__(self,prepare_data, LLUF_update_p1st, LLUF_update_q,LLUF_update_p2nd):
        super().__init__()

        self.prepare_data = prepare_data
        self.LLUF_update_p1st = LLUF_update_p1st
        self.LLUF_update_q = LLUF_update_q
        self.LLUF_update_p2nd = LLUF_update_p2nd

        print(' velocity verletx ')

        #assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    # q_input_list [phi0,phi1,phi2,...] -- over time points
    # p_input_list [pi0,pi1,pi2,...]

    def one_step(self,q_input_list,p_input_list,q_pre,p_pre,l_list,tau, method='q'):

        # h_p1st:
        h_p1st = self.LLUF_update_p1st(q_input_list, p_input_list, q_pre) # q_pre for gnn, no direct interaction with h

        h1 = hamiltonian_derivative(h_p1st,q_pre)
        p_cur = p_pre + 0.5 * tau * (-h1)

        p_input_cur = self.prepare_data.prepare_p_feature_input(q_pre,p_cur,l_list)
        p_input_list.append(p_input_cur)
        p_input_list.pop(0)  # remove first element

        # h_q:
        h_q = self.LLUF_update_q(q_input_list, p_input_list, q_pre)

        h2 = hamiltonian_derivative(h_q,p_cur)
        q_cur1 = q_pre + tau * h2
        q_cur = pbc(q_cur1, l_list) # Changed

        q_input_next = self.prepare_data.prepare_q_feature_input(q_cur, l_list)
        q_input_list.append(q_input_next)
        q_input_list.pop(0)

        # h_p2nd:
        h_p2nd = self.LLUF_update_p2nd(q_input_list, p_input_list, q_cur)

        h3 = hamiltonian_derivative(h_p2nd,q_cur)
        p_cur2 = p_cur + 0.5 * tau * (-h3)

        p_input_cur2 = self.prepare_data.prepare_p_feature_input(q_cur,p_cur2,l_list)
        p_input_list.pop(-1) # Remove last element
        p_input_list.append(p_input_cur2)

        if method == 'q':
            pass
        elif method == 'p':
            h1 = hamiltonian_derivative(h_p1st, p_pre)

        assert_nan(p_cur2)
        assert_nan(q_cur)
        return q_input_list,p_input_list,q_cur,p_cur2,l_list, h_p1st, h_q, h_p2nd, h1, h2, h3 # Last 7 outputs added by me


    def nsteps(self,q_input_list,p_input_list,q_pre,p_pre,l_list,tau,method='q'):

        #assert(n_chain==1),'MD/velocity_verletx,py: error only n_chain = 1 is implemented '

        # our mbpw-net model chain up to predict the new configuration for n-times
        q_input_list,p_input_list,q_cur,p_cur2,l_list, h_p1st_sum, h_q_sum, h_p2nd_sum, h1, h2, h3 = \
                                  self.one_step(q_input_list,p_input_list,q_pre,p_pre,l_list,tau, method=method) # Added

        return q_input_list,p_input_list,q_cur,p_cur2, l_list, h_p1st_sum, h_q_sum, h_p2nd_sum, h1, h2, h3  # Last 7 outputs added
