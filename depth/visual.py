import torch
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from network import *


# Save Images using matplotlib
def imshow(img, mask, frame_img, count=0):
    img = img.cpu()
    mask = mask.detach().cpu()
    # Directory to save the frames
    frame = frame_img + "frame" + str(count) + ".jpg"
    resize = transforms.Resize((360, 640))
    mask = resize(mask)
    img = resize(img)
    mask = mask.squeeze(0)
    img = img / 2 + 0.5
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[0].permute(1, 2, 0), cmap='gray')
    axs[1].imshow(mask[0], cmap='hot', alpha=0.9)
    plt.savefig(frame, bbox_inches='tight', pad_inches=0)
    # plt.show()


def prediction(img, model, frame_dir, device, count):
    """
    Makes predictions on the frame
    :param img: input frame
    :param model: Model needed for prediction
    :param frame_dir:Directory to save the frames
    :param device:Uses GPU or CPU
    :param count: frame counter
    :return:
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ])
    # preprocess the image
    img = preprocess(img).unsqueeze(0)
    img = img.to(device)
    # Make a prediction for the mask
    with torch.no_grad():
        output = model(img)
        imshow(img, output, frame_dir, count)


def save_frames(video_dir, save_dir):
    """
    :param video_dir:Directory of mp4 video
    :param save_dir: Directory to save frames
    :return:
    """
    count = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = URes().to(device)
    model.load()
    model.eval()
    video = cv2.VideoCapture(video_dir)
    success, image = video.read()
    while success:
        prediction(image, model, save_dir, device, count)
        success, image = video.read()
        count += 1


if __name__ == '__main__':
    save_dir = "D:\\VIDEOS\\DEPTH\\"
    video_dir = "D:\\VIDEOS\\default_driving.mp4"
    save_frames(video_dir, save_dir)
