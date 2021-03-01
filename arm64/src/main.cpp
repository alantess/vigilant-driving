#include "camera.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

int main() {
  model m = {"segnet.pt"};
  torch::jit::Module segnet;
  segnet = load_model(&m);
  /* torch::Tensor example = torch::ones({1, 3, 256, 256}, torch::kCUDA); */
  /* std::vector<torch::jit::IValue> inputs; */
  /* inputs.push_back(example); */
  /* torch::Tensor output = net_model.forward(inputs).toTensor(); */
  start_camera(segnet);
}
