import argparse
from torch import nn

from rlkit.torch.vae.iodine import IodineVAE

from rlkit.torch.conv_networks import BroadcastCNN
import rlkit.torch.vae.iodine as iodine
from rlkit.torch.vae.refinement_network import RefinementNetwork
from rlkit.torch.vae.physics_network import PhysicsNetwork
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
import rlkit.torch.vae.conv_vae as conv_vae
from rlkit.torch.vae.unet import UNet
from rlkit.torch.vae.iodine_trainer import IodineTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.launchers.launcher_util import run_experiment
# from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.core import logger
import numpy as np
from scipy import misc
import h5py
import os

class Preprocessor():
    def __init__(self, t_sample, n_frames, total_samples):
        self.t_sample = t_sample
        self.n_frames = n_frames
        self.total_samples = total_samples

    def old_preprocess(self, data):
        """
            (61, 10, 64, 64, 3): (T, N, H, W, C)

        """
        data = data.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        imsize = data.shape[-1]
        data = data.reshape((self.n_frames, -1, 3, imsize, imsize)).swapaxes(0, 1)
        # data = data[:self.total_samples, self.t_sample]
        data = data[:, self.t_sample]
        return data

    def preprocess(self, data):
        """
            (61, 10, 64, 64, 3): (T, N, H, W, C)

        """
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 2, 4)  # (T, N, H, W, C) --> (T, N, C, W, H)
        data = np.swapaxes(data, 0, 1)  # (T, N, C, W, H) --> (N, T, C, W, H)
        # data = data[:self.total_samples, self.t_sample]  # (n, T/s, C, W, H)
        data = data[:, self.t_sample]  # (n, T/s, C, W, H)
        return data

def load_dataset(data_path, train=True):
    mode = 'training' if train else 'validation'
    print("Loading {} dataset...".format(mode))
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    elif 'Ball' in data_path:
        # feats = np.array(hdf5_file[mode]['features'])
        # print('{} feats'.format(mode))
        # print(feats.shape)
        # return feats


        return hdf5_file




    elif 'BlocksGeneration' in data_path:
        feats = np.array(hdf5_file[mode]['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 2, 3)
        return data

def extract_max_k_from_fname(fname):
    start = fname.find('n-')+2
    end = fname.find('_m-')
    substring = fname[start:end]
    ks = map(int, substring.split('-'))
    max_k = max(ks)
    return max_k

def extract_nframe_from_fname(fname):
    start = fname.find('nf-')+3
    end = fname.find('.h5')
    substring = fname[start:end]
    nf = int(substring)
    return nf


def train_vae(variant):
    local_root = '/Users/michaelchang/Documents/Researchlink/Berkeley/rnem_plan/DataGeneration'
    remote_root = '/home/mbchang/Documents/research/rnem-plan/DataGeneration'
    root = local_root if variant['local'] else remote_root
    # train_data_h5 = 'Balls_n-3-4-6_m-1_r-4-6_c--1c_ns-10000_nf-31.h5'
    # test_data_h5 = 'Balls_n-3-4-6_m-1_r-4-6_c--1c_ns-10000_nf-31.h5'

    # # debug
    if variant['debug']:

        # train_data_h5 = 'Balls_n-6_m-1_r-3_c--1_ns-10_nf-61.h5'
        # test_data_h5 = 'Balls_n-6_m-1_r-3_c--1_ns-10_nf-61.h5'
        train_data_h5 = 'Balls_n-6_m-1_r-3_c--1_ns-1000_nf-61.h5'
        test_data_h5 = 'Balls_n-6_m-1_r-3_c--1_ns-1000_nf-61.h5'
    else:
        train_data_h5 = 'Balls_n-3-4-6_m-1_r-4-6_c--1_ns-10000_nf-61.h5'
        test_data_h5 = 'Balls_n-3-4-6_m-1_r-4-6_c--1_ns-10000_nf-61.h5'

    file_max_k = extract_max_k_from_fname(train_data_h5)
    assert file_max_k + 1 == variant['vae_kwargs']['K'] or variant['vae_kwargs']['K'] == 1

    train_path = os.path.join(root, train_data_h5)
    test_path = os.path.join(root, test_data_h5)
    print('train path: {}'.format(train_path))
    print('test path: {}'.format(test_path))

    n_frames = extract_nframe_from_fname(train_data_h5)
    T = variant['vae_kwargs']['T']
    K = variant['vae_kwargs']['K']
    # t_sample = np.array([0, 0, 0, 0, 0, 10, 15, 20, 25, 30])
    # t_sample = np.array([0, 0, 0, 34, 34])  # aha
    t_sample = np.array(range(0, T*variant['subsample'], variant['subsample'])) # can consider subsampling
    print(t_sample)

    train_data = load_dataset(train_path, train=True)
    train_data_preprocessor = Preprocessor(t_sample, n_frames, variant['num_train'])
    # train_data = train_data_preprocessor.preprocess(train_data)
    # print('data shape: {}'.format(train_data.shape))

    test_data = load_dataset(test_path, train=False)
    test_data_preprocessor = Preprocessor(t_sample, n_frames, variant['num_test'])
    # test_data = test_data_preprocessor.preprocess(test_data)
    # print('data shape: {}'.format(test_data.shape))

    # variant['test_preprocessor'] = test_data_preprocessor
    # variant['train_preprocessor'] = train_data_preprocessor

    #logger.save_extra_data(info)
    snapshot_dir = logger.get_snapshot_dir()
    print('Logging to {}'.format(snapshot_dir))
    # variant['vae_kwargs']['architecture'] = iodine.imsize64_large_iodine_architecture
    variant['vae_kwargs']['architecture'] = iodine.imsize64_iodine_architecture
    variant['vae_kwargs']['decoder_class'] = BroadcastCNN
    rep_size = variant['vae_kwargs']['representation_size']

    # refinement_net = RefinementNetwork(**iodine.imsize64_large_iodine_architecture['refine_args'],
    #                                    hidden_activation=nn.ELU())
    refinement_net = RefinementNetwork(**iodine.imsize64_iodine_architecture['refine_args'],
                                       hidden_activation=nn.ELU())
    physics_net = None
    if variant['physics']:
        physics_net = PhysicsNetwork(K, rep_size)

    m = IodineVAE(
        **variant['vae_kwargs'],
        refinement_net=refinement_net,
        dynamic=True,
        physics_net=physics_net,

    )
    # print(m)

    m.to(ptu.device)
    t = IodineTrainer(train_data, test_data, 
                        ##########
                        train_data_preprocessor,
                        test_data_preprocessor,
                        ##########
                        m,
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)

        # print(train_data.shape[0]//variant['algo_kwargs']['batch_size'])
        # assert False


        # t.train_epoch(epoch, batches=train_data.shape[0]//variant['algo_kwargs']['batch_size'])
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_vae=True, train=False, record_stats=True, batches=1,
                     save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
                     save_reconstruction=should_save_imgs)
        #if should_save_imgs:
        #    t.dump_samples(epoch)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')  # TODO

def process_args(args):
    if args.debug:
        args.num_train = 50
        args.num_test = 10
    args.use_gpu = not args.local
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-train', type=int, default=10000)
    parser.add_argument('--num-test', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--seqlength', type=int, default=20)
    parser.add_argument('--subsample', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--exp-prefix', type=str, default='iodine-balls-physics')
    args = parser.parse_args()
    args = process_args(args)

    variant = dict(
        vae_kwargs = dict(
            imsize=64,
            representation_size=32,
            input_channels=3,
            decoder_distribution='gaussian_identity_variance',
            beta=1,
            K=args.k,  # MC: will change
            T=args.seqlength,
        ),
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=args.batch_size,
            lr=1e-4,
            log_interval=0,
        ),
        num_epochs=10000,
        algorithm='VAE',
        save_period=10,
        physics=True,
        num_train=args.num_train,
        num_test=args.num_test,
        subsample=args.subsample,
        local=args.local,
        debug=args.debug
    )

    run_experiment(
        train_vae,
        exp_prefix=args.exp_prefix,
        mode='here_no_doodad',
        variant=variant,
        seed=args.seed,
        use_gpu=args.use_gpu,  # Turn on if you have a GPU
    )



