import torch 


# Loads models into a torch script module 
def load_model():
   repo = 'alantess/vigilant-driving:main/1.0.72'
   model = torch.hub.load(repo, 'segnet', pretrained=True)
   model.eval()
   example = torch.randn((1,3,256,256))
   traced = torch.jit.trace(model, example)
   traced.save("models/segnet.pt")





if __name__ == "__main__":
    load_model()

