#include "Camera/models.cc"
#include <iostream>

int main() {
  model m = {"module_1"};
  load_model(&m);
  start_camera();
}
