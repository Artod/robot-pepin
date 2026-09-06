cmake_minimum_required(VERSION 3.16)
project(pepin_base_cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # the board has four A53 cores: -O2 is not optional
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nlohmann_json REQUIRED)  # header-only, Debian nlohmann-json3-dev

add_library(base_bridge_component SHARED src/base_bridge.cpp)
target_compile_features(base_bridge_component PUBLIC cxx_std_17)
target_include_directories(base_bridge_component PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
# Modern targets only, and one target_link_libraries call: ament_target_dependencies mixes
# link signatures on the same target, which CMake refuses.
target_link_libraries(base_bridge_component PUBLIC
  rclcpp::rclcpp
  rclcpp_components::component
  ${geometry_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  tf2_ros::tf2_ros
  nlohmann_json::nlohmann_json)
# Registers pepin::BaseBridge for component containers and generates the stand-alone executable.
rclcpp_components_register_node(base_bridge_component
  PLUGIN "pepin::BaseBridge"
  EXECUTABLE base_bridge)

install(TARGETS base_bridge_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_package()
