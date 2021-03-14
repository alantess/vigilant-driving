import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import torch.nn as nn


# Gather labels from txt
def get_labels(label_path):
    scaler = MinMaxScaler()
    df = pd.read_csv(label_path)
    scaler.fit(df.values)
    data = scaler.transform(df.values)
    return scaler, data


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name
        self.scaler, _ = get_labels(
            "/media/alan/seagate/dataset/commai_speed/train.txt")
        repo = 'alantess/vigilant-driving:main/1.0.75'
        if self.model_name == 'vidresnet':
            self.base = torch.hub.load(repo,
                                       model_name,
                                       pretrained=True,
                                       timesteps=49,
                                       scaler=self.scaler)
        else:
            self.base = torch.hub.load(repo, model_name, pretrained=True)

    def forward(self, x):
        if self.model_name == 'segnet':
            x = self.base(x).squeeze(0).argmax(0)
        elif self.model_name == 'segnetv2':
            x = self.base(x).squeeze(0).argmax(0)
        elif self.model_name == 'vidresnet':
            x = self.base(x)
            x = torch.mean(x)
        return x


# Loads models into a torch script module
def load_model():
    model_names = ['segnet', 'segnetv2', 'vidresnet']
    for model_name in model_names:
        model_path = "models/" + model_name + ".pt"
        optimized_model_path = "models/" + model_name + "_optimized" + ".pt"

        net = Net(model_name)
        net.eval()

        if model_name == 'vidresnet':
            example = torch.randn(1, 3, 49, 256, 256)
        else:
            example = torch.randn(1, 3, 256, 256)

        # Quantized
        if model_name == 'segnet' or model_name == 'segnetv2':
            net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            net_prepared = torch.quantization.prepare(net)
            net_prepared(example)
            net_int8 = torch.quantization.convert(net_prepared)
            net_script_int8 = torch.jit.script(net_int8)
            torch.jit.save(net_script_int8, optimized_model_path)

        else:
            traced = torch.jit.trace(net, example)
            traced.save(model_path)

    print("Models saved.")


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    load_model()
