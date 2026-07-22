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

# License notices must remain available if the install tree is redistributed
# independently of the source workspace. Use an explicit copy because this
# package otherwise installs large model files as absolute symlinks.
install(
  CODE "
    file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}\")
    execute_process(
      COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different
        \"${CMAKE_CURRENT_SOURCE_DIR}/LICENSE\"
        \"${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.build\"
        \"${CMAKE_CURRENT_SOURCE_DIR}/NOTICE\"
        \"${CMAKE_CURRENT_SOURCE_DIR}/README.md\"
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}\"
      COMMAND_ERROR_IS_FATAL ANY
    )
  "
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # ament_copyright cannot classify the dual Apache-2.0 + BSD-3-Clause
  # contribution statement in CONTRIBUTING.md.
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
