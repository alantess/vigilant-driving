#pragma once
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

typedef struct model {
  std::string model_name;

} model;

void start_camera(torch::jit::Module segnet_model);

torch::jit::Module load_model(model *m);
