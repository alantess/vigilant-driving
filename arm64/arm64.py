import torch

if __name__ == "__main__":
    path = "models/quantized_segnetv2.pt"
    model = torch.load(path)
    model.eval()
