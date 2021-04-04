import torch
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import cv2
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
"""
Deqantizes the tensor when needed. 
"""


class QuantSegnet(nn.Module):
    def __init__(self):
        super(QuantSegnet, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        repo = 'alantess/vigilant-driving:main/1.0.8'
        # Loads Models
        self.base = torch.hub.load(repo, 'segnet', pretrained=True)

    def forward(self, x):
        x = self.quant(x)
        x = self.base(x)
        x = self.dequant(x)
        return x


class QuantSegnetV2(nn.Module):
    def __init__(self):
        super(QuantSegnetV2, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        repo = 'alantess/vigilant-driving:main/1.0.8'
        # Loads Models
        self.base = torch.hub.load(repo, 'segnetv2', pretrained=True)

    def forward(self, x):
        x = self.dequant(x)
        x = self.base(x)
        x = self.quant(x)
        return x


class vDevice(object):
    def __init__(self, quantize=False):
        self.quantize = quantize
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

        if quantize:
            self.segNet = QuantSegnet()
            self.segnetv2 = QuantSegnetV2()

        # Tracks each timestep
        self.cur_step = 0
        self.overlay_images = []
        # Normalization
        self.norm, _ = self._get_labels(
            "/media/alan/seagate/dataset/commai_speed/train.txt")
        self.img_size = None
        self.size = 256
        self.timesteps = 49
        # Preprocessing
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
        # Needed for speednet
        self.image_tensor = torch.zeros(
            (1, 3, self.timesteps, self.size, self.size), device=self.device)

        self._eval()
        """
        Holds model for quantization
        Cannot quantize speednet - No support for conv3d
        """
        if self.quantize:
            self._quantize_models(self.segNet)
            self._quantize_models(self.segnetv2)

    # Prints size of model
    def _print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print(' Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    # Static Quantization
    def _quantize_models(self, model, backend='qnnpack', config=False):
        torch.backends.quantized.engine = backend
        qconfig = torch.quantization.get_default_qconfig(backend)
        model.to('cpu')
        model.qconfig = qconfig
        # Needs to be off for conv2 to work
        model.base.base_model.backbone.qconfig = None
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=False)
        self._print_size_of_model(model)

        print("Models Quantized.")

    # Uses camera for predictions
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

    # Segnetv1 (Lanes) - Displays direct/alternative lanes
    def _segnet_pred(self, image):
        """
        input:image
        returns: mask for each segnet model  
        """
        if self.quantize:
            self.device = torch.device('cpu')
        resize = transforms.Resize(self.img_size)
        image = self.preprocess(image)
        image = image.to(self.device, dtype=torch.float32).unsqueeze(0)
        mask = {"v1": self.segNet(image), "v2": self.segnetv2(image)}
        for version in mask:
            mask[version] = mask[version].argmax(1).detach().cpu().mul(
                127.5).clamp(0, 255)
            mask[version] = resize(mask[version]).numpy().squeeze(0)

        return mask

    # Predicts MPH
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

    # Uses video to make predictions
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

    # Sets models to eval mode
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

    # Overlay all of the frames from the model predictions and displays them
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
