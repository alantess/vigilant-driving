import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from network import *


# Gather labels from txt
def get_labels(label_path):
    scaler = StandardScaler()
    df = pd.read_csv(label_path)
    scaler.fit(df.values)
    data = scaler.transform(df.values)
    return scaler, data


# Train model
def train_model(
    train_dir,
    labels_dir,
    transform,
    loss_fn,
    time_steps,
    IMG_SIZE,
    EPOCHS=1,
    load=False,
):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = VideoResNet(time_steps, 1e-6).to(device)
    # Load Model
    if load:
        model.load()
        print("Model loaded.")

    # Retrieve Labels
    _, labels = get_labels(labels_dir)
    labels = torch.tensor(labels, dtype=torch.float, device=device)

    # Enable Grad Scaler
    scaler = torch.cuda.amp.GradScaler()

    best_loss = np.inf
    loss = 0

    print("Starting")
    for epoch in range(EPOCHS):
        # Reset loss
        total_loss, running_loss = 0.0, 0.0
        # Store Frames
        image_tensor = torch.zeros((1, 3, time_steps, IMG_SIZE, IMG_SIZE),
                                   device=device)
        # Gather Video
        video = cv2.VideoCapture(train_dir)
        success, image = video.read()
        index, i, cur_step = 0, 0, 0
        while success:
            # Assign the state vector a frame
            image_tensor[:, :, cur_step, :, :] = transform(image)

            cur_step += 1
            if cur_step % time_steps == 0:
                i += 1
                index = i * time_steps
                # Set labels
                y = labels[index - time_steps:index].reshape(-1)

                # zeros out gradients
                for p in model.parameters():
                    p.grad = None

                # Forward Pass
                with torch.cuda.amp.autocast():
                    pred = model(image_tensor)
                    # Compute loss
                    loss = loss_fn(pred, y)

                total_loss += loss.item()

                if i % 100 == 0:
                    running_loss = total_loss / 100
                    print(f"[{epoch}/{i}] \tLoss {running_loss:.5f} ")

                # Backward Pass
                scaler.scale(loss).backward()
                scaler.step(model.optimizer)
                scaler.update()

                # Reset Frame Tesnor and Current Timestep
                image_tensor = torch.zeros(
                    (1, 3, time_steps, IMG_SIZE, IMG_SIZE), device=device)

                cur_step = 0

            success, image = video.read()
        print(
            f"EPOCH # {epoch}: Loss: {total_loss:.3f}  Best: {best_loss:.3f}")

        # Save model based on loss
        if total_loss < best_loss:
            print("Saving...")
            best_loss = total_loss
            model.save()


# Test Model
def test_model(test_dir, labels_dir, gif_dir, transform, time_steps, IMG_SIZE):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load Model
    model = VideoResNet(time_steps).to(device)
    model.load()

    # Normalization
    norm, _ = get_labels(labels_dir)

    # CV2 Text overlay variables
    font = cv2.FONT_ITALIC
    textLocation = (570, 412)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    mph = "-"

    img_count = 0
    print("Starting...")
    with torch.no_grad():
        # Store Frames
        image_tensor = torch.zeros((1, 3, time_steps, IMG_SIZE, IMG_SIZE),
                                   device=device)
        # Gather Video
        video = cv2.VideoCapture(test_dir)
        success, image = video.read()
        i, cur_step = 0, 0

        while success:
            image_tensor[:, :, cur_step, :, :] = transform(image)
            cur_step += 1
            if cur_step % time_steps == 0:
                i += 1
                # Forward Pass
                pred = model(image_tensor).reshape(time_steps, 1)
                # Get average Speed at S(t)
                mph = norm.inverse_transform(pred.cpu().detach().numpy())
                mph = round(np.mean(mph))

                # Reset State Tensors, and current timestep
                image_tensor = torch.zeros(
                    (1, 3, time_steps, IMG_SIZE, IMG_SIZE), device=device)
                cur_step = 0
            # Save image
            image = cv2.putText(image, str(mph), textLocation, font, fontScale,
                                fontColor, lineType)

            cv2.imshow('main', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            img_count += 1
            success, image = video.read()
