import models.archs.CWNet as CWNet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    nc = opt_net['nc']
    num_block = opt_net['num_block']

    if which_model == 'CWNet':
        netG = CWNet.CWNet(num_block=num_block)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

