#include "models.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280

// Loads the model
void load_model(model *m) {
  std::cout << "LOADING MODEL: " << m->model_name << std::endl;
}

// Opens the Camera
int start_camera() {
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
  for (;;) {
    cap.read(frame);
    /* torch::Tensor frame_tensor = torch::tensor(frame); */
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
