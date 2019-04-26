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

def load_dataset(data_path, train=True):
    mode = 'training' if train else 'validation'
    print("Loading {} dataset...".format(mode))
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    elif 'Ball' in data_path:
        feats = np.array(hdf5_file[mode]['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        return data
    elif 'BlocksGeneration' in data_path:
        feats = np.array(hdf5_file[mode]['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 2, 3)
        return data


def train_vae(variant):
    #train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train.hdf5'
    #test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    # train_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorBigTwoBallSmall.h5'  # MC: will use
    # test_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorBigTwoBallSmall.h5'  # MC: will use

    train_path = '/Users/michaelchang/Documents/Researchlink/Berkeley/rnem_plan/DataGeneration/Balls_n-2_m-1_r-7_c-8_ns-1000.h5'  # MC: will use
    test_path = '/Users/michaelchang/Documents/Researchlink/Berkeley/rnem_plan/DataGeneration/Balls_n-2_m-1_r-7_c-8_ns-1000.h5'  # MC: will use

    # train_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock.h5'
    # test_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock.h5'

    train_data = load_dataset(train_path, train=True)
    test_data = load_dataset(test_path, train=False)

    n_frames = 51
    imsize = train_data.shape[-1]
    T = variant['vae_kwargs']['T']
    K = variant['vae_kwargs']['K']
    rep_size = variant['vae_kwargs']['representation_size']
   # t_sample = np.array([0, 0, 0, 0, 0, 10, 15, 20, 25, 30])
    t_sample = np.array([0, 0, 0, 34, 34])
    train_data = train_data.reshape((n_frames, -1, 3, imsize, imsize)).swapaxes(0, 1)[:1000, t_sample]  # MC: note the 1000
    test_data = test_data.reshape((n_frames, -1, 3, imsize, imsize)).swapaxes(0, 1)[:50, t_sample]
    print('train data shape: {}'.format(train_data.shape))
    print('test_data shape: {}'.format(test_data.shape))
    #logger.save_extra_data(info)
    logger.get_snapshot_dir()
    # variant['vae_kwargs']['architecture'] = iodine.imsize64_large_iodine_architecture
    variant['vae_kwargs']['architecture'] = iodine.imsize64_iodine_architecture
    variant['vae_kwargs']['decoder_class'] = BroadcastCNN

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
    print(m)

    m.to(ptu.device)
    t = IodineTrainer(train_data, test_data, m,
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch, batches=train_data.shape[0]//variant['algo_kwargs']['batch_size'])
        t.test_epoch(epoch, save_vae=True, train=False, record_stats=True, batches=1,
                     save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
                     save_reconstruction=should_save_imgs)
        #if should_save_imgs:
        #    t.dump_samples(epoch)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        vae_kwargs = dict(
            imsize=64,
            representation_size=32,
            input_channels=3,
            decoder_distribution='gaussian_identity_variance',
            beta=1,
            K=3,  # MC: will change
            T=5,
        ),
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=8,
            lr=1e-4,
            log_interval=0,
        ),
        num_epochs=10000,
        algorithm='VAE',
        save_period=5,
        physics=True
    )

    run_experiment(
        train_vae,
        exp_prefix='iodine-balls-physics',
        mode='here_no_doodad',
        variant=variant,
        # use_gpu=True,  # Turn on if you have a GPU
        use_gpu=False,  # Turn on if you have a GPU
    )



