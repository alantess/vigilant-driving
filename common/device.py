import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import cv2


class vDevice(object):
    def __init__(self):
        # Device
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        repo = 'alantess/vigilant-driving:main/1.0.8'
        # Loads Models
        self.segNet = torch.hub.load(repo, 'segnet',
                                     pretrained=True).to(self.device)
        self.speedNet = torch.hub.load(repo,
                                       'vidresnet',
                                       pretrained=True,
                                       timesteps=49).to(self.device)
        self.segnetv2 = torch.hub.load(repo, 'segnetv2',
                                       pretrained=True).to(self.device)

        self.cur_step = 0
        self.overlay_images = []
        self.norm, _ = self._get_labels(
            "/media/alan/seagate/dataset/commai_speed/train.txt")
        self.img_size = None
        self.size = 256
        self.timesteps = 49
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.size, self.size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.inv_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1., 1., 1.]),
        ])
        self.mph = "--"
        self.image_tensor = torch.zeros(
            (1, 3, self.timesteps, self.size, self.size), device=self.device)

        self._eval()

    def init_camera(self, size):
        self.img_size = size[::-1]
        """
        :params --> size: size of the frame. must be a tuple WxH
        """
        cap = cv2.VideoCapture(0)

        while (True):
            _, frame = cap.read()
            frame = cv2.resize(frame, size)

            self._overlay(frame)
            # cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _segnet_pred(self, image):
        """
        input:image
        returns: mask for each segnet model  
        """
        resize = transforms.Resize(self.img_size)
        image = self.preprocess(image)
        image = image.to(self.device, dtype=torch.float32).unsqueeze(0)
        mask = {"v1": self.segNet(image), "v2": self.segnetv2(image)}
        for version in mask:
            mask[version] = mask[version].argmax(1).detach().cpu().mul(
                127.5).clamp(0, 255)
            mask[version] = resize(mask[version]).numpy().squeeze(0)

        return mask

    def _speed_pred(self, image):
        """
        Takes 
        """
        # CV2 Text overlay variables
        font = cv2.FONT_HERSHEY_PLAIN
        textLocation = (50, 50)
        fontScale = 2
        fontColor = (255, 255, 255)
        lineType = 2
        # Preprocess
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.size, self.size)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])

        # Sets Frame at St
        self.image_tensor[:, :, self.cur_step, :, :] = transform(image)
        self.cur_step += 1

        # Makes prediction
        if self.cur_step % self.timesteps == 0:

            # Forward Pass
            pred = self.speedNet(self.image_tensor).reshape(self.timesteps, 1)
            # Get average Speed at S(t)
            mph = self.norm.inverse_transform(pred.cpu().detach().numpy())
            # Get average Speed at S(t)
            mph = round(np.mean(mph))
            self.mph = mph
            # Reset Tensor
            self.image_tensor = torch.zeros(
                (1, 3, self.timesteps, self.size, self.size),
                device=self.device)
            self.cur_step = 0

        image = cv2.putText(image, str(self.mph), textLocation, font,
                            fontScale, fontColor, lineType)

        return image

    def init_video(self, video_dir, size):
        self.img_size = size[::-1]
        video = cv2.VideoCapture(video_dir)
        success, frame = video.read()
        while success:
            success, frame = video.read()
            frame = cv2.resize(frame, size)
            self._overlay(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def _eval(self):
        self.speedNet.eval()
        self.segNet.eval()
        self.segnetv2.eval()

    # Gather labels from txt
    def _get_labels(self, label_path):
        """
        Normalizes MPH
        """
        scaler = StandardScaler()
        df = pd.read_csv(label_path)
        scaler.fit(df.values)
        data = scaler.transform(df.values)
        return scaler, data

    def _overlay(self, image):
        """
        Ovelay outputs from models
        """

        image = self._speed_pred(image)
        # Segnets
        segnet = self._segnet_pred(image)

        for x in segnet:
            segnet[x] = np.stack((segnet[x], ) * 3, axis=-1).astype(np.uint8)
            segnet[x] = cv2.applyColorMap(segnet[x], cv2.COLORMAP_JET)

        imagev1 = cv2.addWeighted(image, 0.9, segnet["v1"], 0.2, 0)
        imagev2 = cv2.addWeighted(image, 0.9, segnet["v2"], 0.2, 0)
        # Concat and resize frames
        final = cv2.hconcat([imagev1, imagev2])
        final = cv2.resize(final, self.img_size[::-1])

        cv2.imshow('Main', final)
