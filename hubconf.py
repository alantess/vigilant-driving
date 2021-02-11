dependencies = ["torch","captum"]


import torch

from segnet.network import SegNet
from speed.main import VidResnet
from depth.network import DisparityNet, URes

def segnet(pretrained=False, **kwargs):
    """
    :param pretrained: Loads the model weights.
    :param chkpt_dir: Where the model is saved.
    """

    model = SegNet(**kwargs)

    if pretrained:
        checkpoint = "https://github.com/alantess/vigilant-driving/releases/download/1.0.2/deeplab_weights_driving"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model

def vidresnet(pretrained=False, **kwargs):
    """
    :param Loads model weights
    :param Represents the time steps
    :param lr Learning rate
    :param chkpt_dir directory for where the model is saved
    """
    model = VidResnet(**kwargs)

    if pretrained:
        checkpoint = "https://github.com/alantess/vigilant-driving/releases/download/1.0.2/resvid_net_gru_weights"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model

def ures(pretrained=False, **kwargs):
    """

    :param pretrained:Loads model weights
    :param n_channels: How many output channel for the image output
    :param chkpt_dir: Directory where the model is saved
    """
    model = URes(**kwargs)
    if pretrained:
        checkpoint = "https://github.com/alantess/vigilant-driving/releases/download/1.0.2/resnet_encoder_decoder_2"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model

def disparitynet(pretrained=False, **kwargs):
    """
    :param pretrained: Loads model weights
    :param chkpt_dir:Directory where the model is saved
    """

    model = DisparityNet(**kwargs)
    if pretrained:
        checkpoint = "https://github.com/alantess/vigilant-driving/releases/download/1.0.2/disparity_net"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model

