#include "models.h"
#include <iostream>

int main() {
  model m = {"module_1"};
  load_model(&m);
  start_camera();
}
