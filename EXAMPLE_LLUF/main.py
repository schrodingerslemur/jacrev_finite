# import torch
# import os
# import numpy as np

# from torchviz import make_dot

# from ML.trainer.trainer import trainer
# import sys
# sys.path.append('../')
# from utils import utils, check_param_dict
# from LLUF_june4.utils.system_logs import system_logs
# from LLUF_june4.utils.mydevice import mydevice
# from LLUF_june4.data_loader.data_loader import data_loader
# from LLUF_june4.data_loader.data_loader import my_data

from torch.func import jacrev
from JacrevFinite import JacrevFinite
from LLUF_class import LLUF_class

# def main():
#     _ = mydevice()
#     _ = system_logs(mydevice)
#     system_logs.print_start_logs()

#     torch.set_default_dtype(torch.float64)
#     #torch.autograd.set_detect_anomaly(True)

#     torch.manual_seed(34952)
#     np.random.seed(34952)

#     argv = sys.argv
#     if len(argv) != 18:
#         print(
#             'usage <programe> <loadfile> <net type> <single net type> <multi net type> <readout net type> <trans layer> <gnn layer> <nnode> <tau_long> <window sliding> <batchsize> <ngrid> <b> <a> <nitr> <saved epoch> <start epoch> <filename>')
#         quit()

#     net_type = argv[1]
#     single_parnet_type = argv[2]
#     multi_parnet_type = argv[3]
#     readout_net_type = argv[4]
#     trans_layer = int(argv[5])
#     gnn_layer = int(argv[6])
#     nnode = int(argv[7])
#     tau_long = float(argv[8])
#     window_sliding = int(argv[9])
#     batch_size = int(argv[10])
#     ngrid = int(argv[11])
#     b = argv[12].split(',')
#     a = argv[13].split(',')
#     nitr = int(argv[14])
#     loss_weights = argv[15].split(',')
#     start_epoch = int(argv[16])
#     loadfile = argv[17]

#     for i in range(len(loss_weights)):
#         if isinstance(loss_weights[i], float):
#             loss_weights[i] = float(loss_weights[i])
#         else:
#             loss_weights[i] = eval(loss_weights[i])

#     for i in range(len(b)):
#         if isinstance(b[i], float):
#             b[i] = float(b[i])
#         else:
#             b[i] = eval(b[i], {'np': np})

#     for i in range(len(a)):

#         if isinstance(a[i], float):
#             a[i] = float(a[i])
#         else:
#             a[i] = eval(a[i], {'np': np})

#     if loadfile.strip() == "None":
#         loadfile = None
#     else:
#         loadfile = loadfile.strip()

#     traindict = {"loadfile": loadfile,  # to load previously trained model
#                  "net_nnodes": nnode,  # number of nodes in neural nets
#                  "pw4mb_nnodes": 128,  # number of nodes in neural nets
#                  "pw_output_dim": 2,
#                  "init_weights": 'tanh',  #relu
#                  "optimizer": 'Adam',
#                  "single_particle_net_type": single_parnet_type,
#                  "multi_particle_net_type": multi_parnet_type,
#                  "readout_step_net_type": readout_net_type,
#                  "n_encoder_layers": trans_layer,
#                  "n_gnn_layers": gnn_layer,
#                  "edge_attention": True,
#                  "d_model": 256,
#                  "nhead": 8,
#                  "net_dropout": 0.0,  # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
#                  "grad_clip": 0.5,  # clamp the gradient for neural net parameters
#                  "tau_traj_len": 8 * tau_long,  # n evaluations in integrator
#                  "tau_long": tau_long,
#                  "loss_weights": loss_weights,
#                  "window_sliding": window_sliding,  # number of times to do integration before cal the loss
#                  "ngrids": ngrid,  # 6*len(b_list)
#                  "b_list": b,  #   # grid lattice constant for multibody interactions
#                  "a_list": a,  #[np.pi/8], #
#                  "maxlr": 1e-4,  # starting learning rate # HK
#                  "tau_init": 1,  # starting learning rate
#                  }

#     window_sliding_list = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 16]

#     if traindict["window_sliding"] not in window_sliding_list:
#         print('window_sliding is not valid, need ', window_sliding_list)
#         quit()

#     lossdict = {"polynomial_degree": 4,
#                 "rthrsh": 0.7,
#                 "e_weight": 1,
#                 "reg_weight": 10} # 10

#     data = {"train_file": 'data_sets\\n16lt0.1stpstraj24_s100.pt',
#             "valid_file": 'data_sets\\n16lt0.1stpstraj24_s100.pt',
#             "test_file": 'data_sets\\n16lt0.1stpstraj24_s100.pt',
#             #"train_pts" : 1800000,
#             #"vald_pts"  : 200000,
#             #"test_pts"  : 200000,
#             "train_pts": 1,
#             "vald_pts": 3,
#             "test_pts": 3,
#             "batch_size": batch_size,
#             "window_sliding": traindict["window_sliding"]}

#     maindict = {"start_epoch": start_epoch,
#                 "end_epoch": 1,
#                 #"save_dir"        : './results20231113/traj_len08ws0{}tau{}ngrid{}{}_dpt1800000'.format(window_sliding,traindict["tau_long"],ngrid,net_type),
#                 "save_dir": './results20240603',
#                 #traj_len08ws0{}tau{}ngrid{}{}_dpt180000'.format(window_sliding,traindict["tau_long"],ngrid,net_type),
#                 "tau_short": 1e-4,
#                 "nitr": nitr,  # for check md trajectories
#                 "append_strike": nitr,  # for check md trajectories
#                 "ckpt_interval": 100,  # for check pointing
#                 "val_interval": 1,  # no use of valid for now
#                 "verb": 1}  # peroid for printing out losses

#     utils.print_dict('trainer', traindict)
#     utils.print_dict('loss', lossdict)
#     utils.print_dict('data', data)
#     utils.print_dict('main', maindict)

#     print('begin ------- check param dict -------- ', flush=True)
#     check_param_dict.check_maindict(traindict)
#     check_param_dict.check_datadict(data)
#     check_param_dict.check_traindict(maindict, traindict["tau_long"])
#     print('end   ------- check param dict -------- ')

#     data_set = my_data(data["train_file"], data["valid_file"], data["test_file"],
#                        traindict["tau_long"], traindict["window_sliding"], traindict["tau_traj_len"],
#                        data["train_pts"], data["vald_pts"], data["test_pts"])
#     loader = data_loader(data_set, data["batch_size"])

#     # utils.check_data(loader,data_set,traindict["tau_traj_len"],
#     #           traindict["tau_long"],maindict["tau_short"],
#     #           maindict["nitr"],maindict["append_strike"])

#     train = trainer(traindict, lossdict)

#     train.load_models()

#     print('begin ------- initial learning configurations -------- ')
#     train.verbose(0, 'init_config')
#     print('end  ------- initial learning configurations -------- ')

    # for e in range(maindict["start_epoch"], maindict["end_epoch"]):

    #     cntr = 0
    #     for qpl_input, qpl_label in loader.train_loader:

# ------------------------------------------------IMPLEMENTATION----------------------------------------------------------------------

            mydevice.load(qpl_input)
            q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)

            # q_traj,p_ttaj [traj,nsamples,nparticles,dim]
            # q_label,p_label,l_init [nsamples,nparticles,dim]

            networks = [train.prepare_data_obj, train.mlvv, l_init]
            lluf = LLUF_jacrev(networks) 
            q_cur, p_cur, q_traj_7, p_traj_7 = lluf.preprocess(q_traj, p_traj)

            # q_traj_7, p_traj_7 [7,1,16,2]
            # q_cur, p_cur [1,1,16,2]
            
            print('-------------------------------------q---------------------------------------')
            jacobian = JacrevFinite(function=lluf.q_network,dim=1,wrapper=lluf.wrapper, num_args=0,delta=1e-5)(q_cur, p_cur, q_traj_7, p_traj_7)
            jacobian = jacobian.sum(dim=(0,1,2,3,4)) # jacobian = [outputdim, inputdim]

            jacobian_actual = jacrev(lluf.q_forward, argnums=0)(q_cur, p_cur, q_traj_7, p_traj_7)
            jacobian_actual = jacobian_actual.sum(dim=(0,1,2,3,4))

            print('----------------JacrevFinite function----------------')
            print(jacobian)
            print('----------------jacobian_actual-----------------------')
            print(jacobian_actual)

            print('mean_error', jacobian_actual.sub(jacobian).abs().mean())
            print('max_error', jacobian_actual.sub(jacobian).abs().max())

            print('------------------------------------p----------------------------------------')
            jacobian = JacrevFinite(function=lluf.p_network,dim=1,wrapper=lluf.wrapper, num_args=1, delta=1e-5)(q_cur, p_cur, q_traj_7, p_traj_7)
            jacobian = jacobian.sum(dim=(0,1,2,3,4)) # jacobian = [outputdim, inputdim]

            jacobian_actual = jacrev(lluf.p_forward, argnums=1)(q_cur, p_cur, q_traj_7, p_traj_7)
            jacobian_actual = jacobian_actual.sum(dim=(0,1,2,3,4))

            print('----------------JacrevFinite function----------------')
            print(jacobian)
            print('----------------jacobian_actual-----------------------')
            print(jacobian_actual)

            print('mean_error', jacobian_actual.sub(jacobian).abs().mean())
            print('max_error', jacobian_actual.sub(jacobian).abs().max())

# -------------------------------------------------------------------------------------------------------------------------------------

#             cntr += 1
#             if cntr % 10 == 0: print('.', end='', flush=True)

        
#         print(cntr, 'batches \n')

#         if e % maindict["verb"] == 0:
#             train.verbose(e + 1, 'train')
#             system_logs.record_memory_usage(e + 1)
#             print('time use for ', maindict["verb"], 'epoches is: ', end='')
#             system_logs.record_time_usage(e + 1)

#         if e % maindict["ckpt_interval"] == 0:
#             filename = './{}/mbpw{:06d}.pth'.format(maindict["save_dir"], e + 1)
#             print('saving file to ', filename)
#             train.checkpoint(filename)

#         if e % maindict["val_interval"] == 0:
#             train.loss_obj.clear()
#             for qpl_input, qpl_label in loader.val_loader:
#                 q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)
#                 train.eval(q_traj, p_traj, q_label, p_label, l_init, maindict['tau_short'])
#             train.verbose(e + 1, 'eval')

#     system_logs.print_end_logs()


# if __name__ == '__main__':
#     # python C:\Users\brend\Dropbox\Python\Research\Research_ref\LLUF_june4\maintrain09.py api0lw8421 transformer_type gnn_identity mlp_type 2 2 128 0.1 8 1 6 0.2 0 1000 0,1/8,0,1/4,0,1/2,0,1 0 results20240603\mbpw000001.pth
#     # usage <programe> <loadfile> <net type> <single net type> <multi net type> <readout net type> <trans layer> <gnn layer> <nnode> <tau_long> <window sliding> <batchsize> <ngrid> <b> <a> <nitr> <saved epoch> <start epoch> <filename>')

#     main()
