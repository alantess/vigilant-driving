#include "camera.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <torch/script.h>
#include <torch/torch.h>

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280

// Loads the model
torch::jit::Module load_model(model *m) {
  std::string directory = "../models/";
  torch::jit::Module module = torch::jit::load(directory + m->model_name);
  module.to(torch::kCUDA);
  std::cout << "Module Loaded: " << m->model_name << std::endl;
  return module;
}

// Makes a prediction
void make_predictions(cv::Mat frame, torch::jit::Module segnet_module) {
  // Input Vector
  std::vector<torch::jit::IValue> inputs;
  // Mean and Standard Deviation
  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> std = {0.229, 0.224, 0.225};
  // Turns frame into tensor and reshapes 1x3x256x256
  torch::Tensor frame_tensor = torch::from_blob(frame.data, {1, 256, 256, 3});
  frame_tensor = frame_tensor.permute({0, 3, 2, 1});
  // Normalization
  frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
  // Sends tensor to GPU
  frame_tensor = frame_tensor.to(torch::kCUDA, torch::kFloat32);
  // Forward pass
  inputs.push_back(frame_tensor);
  torch::Tensor prediction = segnet_module.forward(inputs).toTensor();
}

// Opens the Camera
int start_camera(torch::jit::Module segnet_model) {
  // Camera Variables
  cv::Mat frame;
  cv::VideoCapture cap;
  int deviceId = 0;
  int apiID = cv::CAP_ANY;

  cap.open(deviceId, apiID);

  // Checks to see if camera can be opened
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Can't open camera\n";
    return -1;
  }
  printf("Opening Camera");

  // Set the dimentions
  cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT);
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_FOCUS, 0);

  printf("Press any key to disable camera\n");
  // Reads each frame
  for (;;) {
    cap.read(frame);
    make_predictions(frame, segnet_model);

    if (frame.empty()) {
      std::cerr << "ERROR: blank frame.\n";
      break;
    }
    cv::imshow("Live", frame);
    if (cv::waitKey(5) >= 0) {
      break;
    }
  }
  return 0;
}
