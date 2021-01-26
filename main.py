import argparse
import os
import torch as T
import cv2
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from gtrxl_torch.gtrxl_torch import GTrXL
from sklearn.preprocessing import MinMaxScaler
from torch import optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models


# Model
class VidResnet(nn.Module):
    def __init__(self, n_outs,lr=0.00000011, chkpt_dir="models"):
        super(VidResnet, self).__init__()
        self.base_model = models.video.r3d_18(pretrained=True)
        self.file = os.path.join(chkpt_dir, 'resvid_net_gru_weights')
        self.gru = nn.GRU(400,256,3)
        self.fc1 = nn.Linear(256,128)
        self.output = nn.Linear(128,n_outs)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.base_model(x)
        x = x.unsqueeze(1)
        out, h0 = self.gru(x)
        out = out.view(-1)
        out = F.relu(self.fc1(out))
        out = self.output(out)
        return out

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))

# Display Frames
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Retrieves label and sets scaler
def get_labels(dir):
    df = pd.read_csv(dir)
    scaler = MinMaxScaler()
    scaler.fit(df.values)
    data = scaler.transform(df.values)
    return scaler , data



def vid_to_tensor(dir, transform):
    image_tensor = T.zeros((20400, 3, 256, 256))
    video = cv2.VideoCapture(dir)
    success, image = video.read()
    cur_step = 0
    while success:
        image_tensor[cur_step] = transform(image)
        success, image = video.read()
        cur_step += 1

    return image_tensor


def train(train_dir,labels_dir,  transform,criterion, time_steps, SIZE ,EPOCHS =1):
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    best_loss = np.inf

    print("VIDEO CNN-GRU")
    model = VidResnet(time_steps).to(device)

    # Comment if model hasn't been saved
    model.load()

    print("Model Parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    scaler, labels = get_labels(labels_dir)
    labels = T.tensor(labels, dtype=T.float, device=device)
    
    print("starting...")
    for epoch in range(EPOCHS):
        total_loss , running_loss = 0.0, 0.0
        speed_history = np.array([])
        # State vector 
        image_tensor = T.zeros((1,3,time_steps,SIZE,SIZE), device=device)
        # Retrieve the video
        video = cv2.VideoCapture(train_dir)
        success, image = video.read()
        index,i,cur_step  = 0,0,0

        while success: 
            # Assign the state vector a frame
            image_tensor[:,:,cur_step,:,:] = transform(image)

            cur_step += 1
            if cur_step % time_steps == 0: 
                i += 1
                index = i * time_steps
                # Set labels
                y = labels[index-time_steps:index].reshape(-1)
                
                # zeros out gradients
                for p in model.parameters():
                    p.grad = None

                # Make a predictions and get the loss
                pred = model(image_tensor)
                # Takes the predictions and returns them into MPH format
                prediction = pred.reshape(time_steps,1)
                prediction = scaler.inverse_transform(prediction.cpu().detach().numpy())
                speed_history= np.append(speed_history, prediction.flatten())

                loss = criterion(pred,y)
                loss.backward()
                # Optimizer step
                model.optimizer.step()
                
                # Calculate loss
                running_loss += loss.item()
                total_loss += running_loss
                if i % 255 == 254: 
                    running_loss /= 255
                    print(f"[{epoch}/{i+1}] \tLoss {running_loss:.5f} ")
                
                # Reset State Tensors, and current timestep 

                image_tensor = T.zeros(( 1,3,time_steps,SIZE,SIZE), device=device)

                cur_step = 0
            success, image = video.read()


        print("Total loss: ", total_loss )

        if total_loss < best_loss:
            print("Saving...")
            best_loss = total_loss
            # Uncomment to save the model
            # model.save()

        # Saving the predictions 
        np.savetxt("train_pred.txt", speed_history)
        print("Predictions Saved")

    # Uncomment to show frames
    # imshow(torchvision.utils.make_grid(image_tensor.cpu()))


def test(test_dir,labels_dir ,transform,time_steps,SIZE):
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    speed_history = np.array([])

    model = VidResnet(time_steps).to(device)
    model.load()
    print("Model has been loaded.")

    print("Model Parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    scaler,_ = get_labels(labels_dir)

     
    print("Starting...")
    with T.no_grad(): 
            # State vector 
            image_tensor = T.zeros((1,3,time_steps,SIZE,SIZE), device=device)
            # Start Video
            video = cv2.VideoCapture(test_dir)
            # Read frames from video
            success, image = video.read()
            
            index,i,cur_step  = 0,0,0

            while success: 
                # Assign the state vector a frame
                image_tensor[:,:,cur_step,:,:] = transform(image)

                cur_step += 1
                if cur_step % time_steps == 0: 
                    i += 1
                    index = i * time_steps
                    # Set labels
                    
                    # zeros out gradients
                    for p in model.parameters():
                        p.grad = None

                    # Make a predictions and get the loss
                    pred = model(image_tensor)

                    # Takes the predictions and returns them into MPH 
                    prediction = pred.reshape(time_steps,1)
                    prediction = scaler.inverse_transform(prediction.cpu().detach().numpy())
                    speed_history= np.append(speed_history, prediction.flatten())

                    # Reset State Tensors, and current timestep 
                    image_tensor = T.zeros(( 1,3,time_steps,SIZE,SIZE), device=device)
                    cur_step = 0
                success, image = video.read()



    # Saving the predictions in a txt file
    np.savetxt("test_pred.txt", speed_history)
    print("Finished.\tPrediction saved.")

        # imshow(torchvision.utils.make_grid(image_tensor.cpu()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=bool, default=False)
    parser.add_argument("-epochs", type=int, default = 1)
    args = parser.parse_args()

    train_video = "/mnt/d/pytorch/speedchallenge/data/train.mp4"
    test_video = "/mnt/d/pytorch/speedchallenge/data/test.mp4"
    train_labels = "/mnt/d/pytorch/speedchallenge/data/train.txt"

    SEED = 98
    T.manual_seed(SEED)
    np.random.seed(SEED)
    T.backends.cudnn.benchmark = True
    T.cuda.empty_cache()
    best_loss = np.inf
    # 100 TIMESTEPS = 5 Seconds
    TIMESTEPS = 40
    EPOCHS = args.epochs
    device =  T.device("cuda") 
    SIZE = 112

    criterion = nn.MSELoss()

    transform = transforms.Compose([                   
            transforms.ToTensor(),
            transforms.Resize((SIZE,SIZE)), 
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])

        ])

    
    
    if args.train:
        print("Train Mode.")
        train(train_video,train_labels,transform, criterion, TIMESTEPS, SIZE,EPOCHS)
    else:
        print("Test Mode.")
        test(test_video, train_labels, transform, TIMESTEPS, SIZE)


