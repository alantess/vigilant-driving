import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import cv2


class vDevice(object):
    def __init__(self):
        # Device
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        repo = 'alantess/vigilant-driving:main/1.0.75'
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

    def init_camera(self, size):
        self._eval()
        self.img_size = size[::-1]
        """
        :params --> size: size of the frame. must be a tuple WxH
        """
        cap = cv2.VideoCapture(0)

        while (True):
            _, frame = cap.read()
            if size:
                frame = cv2.resize(frame, size)

            self._overlay(frame)
            cv2.imshow('frame', frame)

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
            mask[version] = mask[version].argmax(1).detach().cpu()
            mask[version] = resize(mask[version]).numpy().squeeze(0)

        return mask

    def _speed_pred(self, image):
        """
        Takes 
        """
        # CV2 Text overlay variables
        font = cv2.FONT_ITALIC
        textLocation = (570, 412)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        mph = "-"
        # Preprocess
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.size, self.size)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])

        # Holds frames at St
        image_tensor = torch.zeros(
            (1, 3, self.timesteps, self.size, self.size), device=self.device)
        # Sets Frame at St
        image_tensor[:, :, self.cur_step, :, :] = transform(image)
        self.cur_step += 1

        # Makes prediction
        if self.cur_step == self.timesteps:
            pred = self.speedNet(image_tensor).reshape(self.timesteps, 1)
            # Get average Speed at S(t)
            mph = self.norm.inverse_transform(pred.cpu().detach().numpy())
            mph = round(np.mean(mph))
            # Overlay speed on image
            cv2.putText(image, str(mph), textLocation, font, fontScale,
                        fontColor, lineType)

            # Reset Tensor
            image_tensor = torch.zeros(
                (1, 3, self.timesteps, self.size, self.size),
                device=self.device)

        return image

    def _eval(self):
        self.speedNet.eval()
        self.segNet.eval()
        self.segnetv2.eval()

    # Gather labels from txt
    def _get_labels(self, label_path):
        """
        Normalizes MPH
        """
        scaler = MinMaxScaler()
        df = pd.read_csv(label_path)
        scaler.fit(df.values)
        data = scaler.transform(df.values)
        return scaler, data

    def _overlay(self, image):
        """
        Ovelay outputs from models
        """
        segnet = self._segnet_pred(image)
        for x in segnet:
            segnet[x] = np.stack((segnet[x], ) * 3, axis=-1)
            image = cv2.addWeighted(image,
                                    0.5,
                                    segnet[x],
                                    0.1,
                                    0,
                                    dtype=cv2.CV_32F)


if __name__ == "__main__":
    v = vDevice()

    v.init_camera(size=(1280, 720))
