#include "camera.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

int main() {
  model m = {"segnet_optimized_arm64.pt"};
  torch::jit::Module segnet;
  segnet = load_model(&m);
  start_camera(segnet);
}
