from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    print("ONE")
    gpus = [0,1,2,3]
    conf = video_64_autoenc()
    train(conf, gpus=gpus)
'''
    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    print("TWO")
    gpus = [0,1,2,3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    print("THREE")
    gpus = [0]
    conf = shanghai_autoenc_latent()
    conf.latent_infer_path = f'checkpoints/shanghai_autoenc/latent.pkl'
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    print("FOUR")
    gpus = [0,1,2,3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval')
'''
