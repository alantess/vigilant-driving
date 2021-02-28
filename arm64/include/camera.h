#pragma once
#include "torch/torch.h"
#include <iostream>

typedef struct model {
  std::string model_name;

} model;

int start_camera();

void load_model(model *m);
