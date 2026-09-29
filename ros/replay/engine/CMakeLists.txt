cmake_minimum_required(VERSION 3.16)
project(pepin_replay)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(costmap_replay src/costmap_replay.cpp)
target_compile_features(costmap_replay PUBLIC cxx_std_17)
target_link_libraries(costmap_replay
  rclcpp::rclcpp
  rosbag2_cpp::rosbag2_cpp
  nav2_costmap_2d::nav2_costmap_2d_core
  nav2_costmap_2d::layers
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  tf2::tf2
  tf2_geometry_msgs::tf2_geometry_msgs
  ${tf2_msgs_TARGETS}
  tf2_ros::tf2_ros)

install(TARGETS costmap_replay RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
