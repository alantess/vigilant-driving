from device import vDevice
import argparse

if __name__ == "__main__":

    video = "/media/alan/seagate/dataset/commai_speed/videos/test.mp4"
    parser = argparse.ArgumentParser(description="Vigilant-Driving")
    parser.add_argument('-video', type=str, default=video)
    parser.add_argument('-quantize', type=bool, default=False)
    parser.add_argument('-load', type=str, default="camera")
    args = parser.parse_args()

    if args.quantize:
        v = vDevice(quantize=True)
    else:
        v = vDevice()
    frame_size = (1280, 720)

    if args.load == "video":
        print("Press 'q' to exit.")
        v.init_video(video, frame_size)
    elif args.load == "camera":
        print("Press 'q' to exit.")
        v.init_camera(size=frame_size)
    else:
        print("Invalid argument. Please Select: \n 'camera' or 'video'")
