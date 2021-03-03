import torch 
from torchvision import transforms
import cv2 as cv
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
       super(Net, self).__init__()
       repo = 'alantess/vigilant-driving:main/1.0.72'
       self.base = torch.hub.load(repo, 'segnet', pretrained=True)

    def forward(self,x):
        x = self.base(x).squeeze(0).argmax(0)
        return x



# Loads models into a torch script module 
def load_model():
    net = Net()
    net.eval()
    example = torch.randn(1,3,256,256)
    traced=torch.jit.trace(net,example)
    traced.save("models/seget_lanes.pt")
   # model = torch.hub.load(repo, 'segnet', pretrained=True)
   # model.eval()
   # img = cv.imread(cv.samples.findFile("img.jpg"))
   # # cv.imshow("Display window", img)
   # device = torch.device("cuda")
   # IMAGE_SIZE =256
   # # k = cv.waitKey(0)
   #     # Preprocess for video
   # video_preprocess = transforms.Compose([
   #      transforms.ToTensor(),
   #      transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
   #      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
   #      ])

    
   # model = model.to(device)
   # x = video_preprocess(img)
   # print(x.max())
   # x = x.unsqueeze(0)
   # x = x.to(device)
   # print(x.size())
   # out = model.forward(x).argmax(1).detach().cpu().numpy()
   # print(x.min())
   # import matplotlib.pyplot as plt
   # x = x.cpu()
   # plt.imshow(x[0].permute(1,2,0), cmap='gray')
   # plt.imshow(out.squeeze(0), cmap='jet', alpha=0.2)
   # plt.show()



    
    # if k == ord("s"):
        # cv.imwrite("img.jpg", img)

   # example = torch.randn((1,3,256,256))
   # traced = torch.jit.trace(model, example)
   # traced.save("models/segnet.pt")





if __name__ == "__main__":
    load_model()

