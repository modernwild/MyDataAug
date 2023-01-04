import imp
import os


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_model(conf):

    src_file = os.path.join('networks', conf.network + '.py')
    netimp = imp.load_source('networks', src_file)
    net = netimp.get_net(conf)
    return net


def count_params(net):
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))