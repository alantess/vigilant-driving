import torch
import torch.nn as nn
import copy
import torch.quantization.quantize_fx as quantize_fx
from torch.utils.mobile_optimizer import optimize_for_mobile


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        repo = 'alantess/vigilant-driving:main/1.0.75'
        self.model = torch.hub.load(repo, 'segnet', pretrained=True)

    def forward(self, x):
        x = self.model(x).squeeze(0).argmax(0)
        return x.mul(100).clamp(0, 255)


model_fp = Net()
model_fp.eval()

model_to_quantize = copy.deepcopy(model_fp)
model_to_quantize.eval()
qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
model_to_quantize.eval()
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# calibrate (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)

# Fusion
model_to_quantize = copy.deepcopy(model_fp)
model_fused = quantize_fx.fuse_fx(model_to_quantize)

# Save model
scipted_model = torch.jit.script(model_fused)
scripted_optimized_moodel = optimize_for_mobile(scipted_model)
torch.jit.save(scripted_optimized_moodel, "models/segnet_fx_mobile.pt")
