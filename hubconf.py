dependencies = ["torch", "captum"]
VERSION = "1.0.73"

import torch

from segnet.network import SegNet
from segnet_V2.network import SegNetV2
from speed.network import VideoResNet
from depth.network import DisparityNet, URes


def segnet(pretrained=False, **kwargs):
    """
    :param pretrained: Loads the model weights.
    :param chkpt_dir: Where the model is saved.
    """

    model = SegNet(**kwargs)

    if pretrained:
        checkpoint = f"https://github.com/alantess/vigilant-driving/releases/download/{VERSION}/deeplab_weights_driving"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model


def vidresnet(pretrained=False, **kwargs):
    """
    :param scaler unnormalizes outputs
    :param timesteps Represents the time steps
    :param lr Learning rate
    :param chkpt_dir directory for where the model is saved
    """
    model = VideoResNet(**kwargs)

    if pretrained:
        checkpoint = f"https://github.com/alantess/vigilant-driving/releases/download/{VERSION}/video_resnet_weights"
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
        checkpoint = f"https://github.com/alantess/vigilant-driving/releases/download/{VERSION}/resnet_encoder_decoder_2"
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
        checkpoint = f"https://github.com/alantess/vigilant-driving/releases/download/{VERSION}/disparity_net"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model


def segnetv2(pretrained=False, **kwargs):
    """
    :param pretrained: Loads model weights
    :param chkpt_dir: Directory where the model is saved
    """
    model = SegNetV2(**kwargs)
    if pretrained:
        checkpoint = f"https://github.com/alantess/vigilant-driving/releases/download/{VERSION}/segnet_v2"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        model.load_state_dict(state_dict)

    return model
