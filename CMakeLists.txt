cmake_minimum_required(VERSION 3.22)
project(moveit_pro_sam2)

# Force symlink installation instead of copying files
set(CMAKE_INSTALL_MODE ABS_SYMLINK)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    models
  DESTINATION
    share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
