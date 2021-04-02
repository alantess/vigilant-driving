import torch
from torchvision import transforms
import numpy as np
from train import *

if __name__ == "__main__":
    train_video = "/media/alan/seagate/dataset/commai_speed/videos/train.mp4"
    test_video = "/media/alan/seagate/dataset/commai_speed/videos/test.mp4"
    train_labels = "/media/alan/seagate/dataset/commai_speed/train.txt"
    gif_dir = "/media/alan/seagate/VIDEOS/speed/"

    SEED = 98
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    TIMESTEPS = 49  # 100 Time steps = 5 Seconds
    EPOCHS = 100
    SIZE = 256
    loss_fn = torch.nn.MSELoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((SIZE, SIZE)),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989])
    ])

    # print("Train Mode.")
    # train_model(train_video, train_labels, transform, loss_fn, TIMESTEPS, SIZE,
    # EPOCHS, True)

    print("Test Mode.")
    test_model(test_video, train_labels, gif_dir, transform, TIMESTEPS, SIZE)
