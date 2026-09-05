from models.archs.tttir_arch import TTTIR_enhance

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    nc = opt_net.get('nc') or 32
    num_block = opt_net['num_block']

    if which_model in ('TTTIR', 'TTTIR_enhance', 'CWNet'):
        netG = TTTIR_enhance(nc=nc, num_block=num_block)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
