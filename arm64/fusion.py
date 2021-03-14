import torch
import copy
import torch.quantization.quantize_fx as quantize_fx

repo = 'alantess/vigilant-driving:main/1.0.75'
model_fp = torch.hub.load(repo, 'segnet', pretrained=True)
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
torch.jit.save(scipted_model, "models/segnet_fx.pt")
