from models.archs.TTTIR import TTTIR

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    nc = opt_net.get('nc') or 32
    num_block = opt_net['num_block']

    if which_model in ('TTTIR', 'CWNet'):
        netG = TTTIR(nc=nc, num_block=num_block)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
