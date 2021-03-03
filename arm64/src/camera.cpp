#include "camera.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <torch/script.h>
#include <torch/torch.h>

#define DEFAULT_HEIGHT 1676
#define DEFAULT_WIDTH 1117
#define IMG_SIZE 256

// Loads the model
torch::jit::Module load_model(model *m) {
  std::string directory = "../models/";
  torch::jit::Module module = torch::jit::load(directory + m->model_name);
  module.to(torch::kCUDA);
  module.eval();
  std::cout << "Module Loaded: " << m->model_name << std::endl;
  return module;
}

// Makes a prediction and returns frame
void make_predictions(cv::Mat frame, torch::jit::Module segnet_module) {
  double alpha = 0.5;
  double beta = (1 - alpha);
  cv::Mat x_frame;
  cv::Mat dst;
  // Mean and Standard Deviation
  std::vector<double> mean = {0.406, 0.456, 0.485};
  std::vector<double> std = {0.225, 0.224, 0.229};
  // Resize
  cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  x_frame = frame;
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);

  // CV2 -> Tensor
  torch::Tensor frame_tensor =
      torch::from_blob(frame.data, {1, IMG_SIZE, IMG_SIZE, 3});

  frame_tensor = frame_tensor.permute({0, 3, 1, 2});
  // Normalize
  frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
  // Set to Device
  frame_tensor = frame_tensor.to(torch::kCUDA);
  // Forward Pass
  std::vector<torch::jit::IValue> input;
  input.push_back(frame_tensor);
  auto pred = segnet_module.forward(input).toTensor().unsqueeze(2).detach().to(
      torch::kCPU);
  // Reset Pixels
  pred = pred.mul(100).clamp(0, 255).to(torch::kU8);
  auto sizes = pred.sizes();
  // Pytorch Tensor to CV2 image
  cv::Mat output_mat(cv::Size{IMG_SIZE, IMG_SIZE}, CV_8UC1, pred.data_ptr());
  cv::cvtColor(output_mat, output_mat, cv::COLOR_GRAY2RGB);
  cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_JET);
  // Combine Original Image with Predicted image
  cv::addWeighted(x_frame, alpha, output_mat, beta, 0.0, dst);
  // RESIZE image and display it
  cv::resize(dst, dst, cv::Size(DEFAULT_HEIGHT, DEFAULT_WIDTH));

  cv::imshow("Display window", dst);
  int k = cv::waitKey(0); // Wait for a keystroke in the window
}

// Opens the Camera
int start_camera(torch::jit::Module segnet_model) {
  // Camera Variables
  cv::Mat frame;
  cv::VideoCapture cap;
  int deviceId = 0;
  int apiID = cv::CAP_ANY;

  std::string image_path = cv::samples::findFile("../img.jpg");
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }

  /* cv::Mat img_pred = make_predictions(img, segnet_model); */
  make_predictions(img, segnet_model);
  /* break; */

  cap.open(deviceId, apiID);

  // Checks to see if camera can be opened
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Can't open camera\n";
    return -1;
  }
  printf("Opening Camera");

  // Set the dimentions 1280x720, Remove AutoFocus/Focus
  cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT);
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_FOCUS, 0);

  printf("Press any key to disable camera\n");
  // Reads each frame
  for (;;) {
    cap.read(frame);

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
