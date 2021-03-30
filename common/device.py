import torch
import cv2


class vDevice(object):
    def __init__(self):
        # Device
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        repo = 'alantess/vigilant-driving:main/1.0.75'
        # Loads Models
        self.disparityNet = torch.hub.load(repo,
                                           'disparitynet',
                                           pretrained=True).to(self.device)
        self.segNet = torch.hub.load(repo, 'segnet',
                                     pretrained=True).to(self.device)
        self.speedNet = torch.hub.load(repo,
                                       'vidresnet',
                                       pretrained=True,
                                       timesteps=49).to(self.device)
        self.segnetv2 = torch.hub.load(repo, 'segnetv2',
                                       pretrained=True).to(self.device)

    def init_camera(self, size=None):
        """
        :params --> size: size of the frame. must be a tuple WxH
        """
        cap = cv2.VideoCapture(0)

        while (True):
            _, frame = cap.read()
            if size:
                frame = cv2.resize(frame, size)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
