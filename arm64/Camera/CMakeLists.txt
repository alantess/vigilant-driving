cmake_minimum_required(VERSION 3.2)





add_library(camera models.cc models.h)

# Problem HERE Won't Load library 
# target_link_libraries(camera PRIVATE ${TORCH_LIBRARIES})




