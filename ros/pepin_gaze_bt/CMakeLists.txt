cmake_minimum_required(VERSION 3.16)
project(pepin_gaze_bt)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(std_srvs REQUIRED)

# bt_navigator loads it by name (plugin_lib_names: pepin_ask_gaze_bt_node, nav.launch.py).
add_library(pepin_ask_gaze_bt_node SHARED src/ask_gaze.cpp)
target_compile_features(pepin_ask_gaze_bt_node PUBLIC cxx_std_17)
# BT_REGISTER_NODES exports the plugin's entry point only when this is defined.
target_compile_definitions(pepin_ask_gaze_bt_node PRIVATE BT_PLUGIN_EXPORT)
target_link_libraries(pepin_ask_gaze_bt_node PUBLIC
  rclcpp::rclcpp
  behaviortree_cpp::behaviortree_cpp
  ${std_srvs_TARGETS})

install(TARGETS pepin_ask_gaze_bt_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_package()
