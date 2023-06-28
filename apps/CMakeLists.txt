cmake_minimum_required(VERSION 3.5)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(OpenCV 4 REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_BINARY_DIR}
        ${DefSLAM_LIB_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        )

set(PROGRAM_NAME hamlyn)
add_executable(${PROGRAM_NAME} hamlyn.cc)
target_link_libraries(${PROGRAM_NAME} ${OpenCV_LIBS} absl::log absl::log_flags absl::flags_parse absl::check absl::status NR-SLAM)

set(PROGRAM_NAME endomapper)
add_executable(${PROGRAM_NAME} endomapper.cc)
target_link_libraries(${PROGRAM_NAME} ${OpenCV_LIBS} absl::log absl::log_flags absl::flags_parse absl::check absl::status NR-SLAM)

set(PROGRAM_NAME simulation)
add_executable(${PROGRAM_NAME} simulation.cc)
target_link_libraries(${PROGRAM_NAME} ${OpenCV_LIBS} absl::log absl::log_flags absl::flags_parse absl::check absl::status NR-SLAM)
