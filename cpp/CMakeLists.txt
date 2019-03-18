cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(infer)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(infer infer.cpp)
target_link_libraries(infer ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET infer PROPERTY CXX_STANDARD 11)