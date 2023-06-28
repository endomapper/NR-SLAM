cmake_minimum_required(VERSION 3.23)

find_package(OpenCV 4 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(MLPACK REQUIRED)
find_package(Pangolin REQUIRED)

set(LIBRARY_NAME NR-SLAM)

# Current library
set(root ${CMAKE_CURRENT_SOURCE_DIR})

set(HEADER_FILES
        ${root}/calibration/camera_model.h
        ${root}/calibration/kannala_brandt_8.h
        ${root}/calibration/pin_hole.h
        ${root}/datasets/endomapper.h
        ${root}/datasets/hamlyn.h
        ${root}/datasets/simulation.h
        ${root}/features/feature.h
        ${root}/features/shi_tomasi.h
        ${root}/map/frame.h
        ${root}/map/keyframe.h
        ${root}/map/map.h
        ${root}/map/mappoint.h
        ${root}/map/regularization_graph.h
        ${root}/map/temporal_buffer.h
        ${root}/mapping/mapping.h
        ${root}/masking/bright_filter.h
        ${root}/masking/border_filter.h
        ${root}/masking/masker.h
        ${root}/masking/predefined_filter.h
        ${root}/matching/lucas_kanade_tracker.h
        ${root}/optimization/g2o_optimization.h
        ${root}/optimization/landmark_vertex.h
        ${root}/optimization/position_regularizer.h
        ${root}/optimization/position_regularizer_with_deformation.h
        ${root}/optimization/reprojection_error.h
        ${root}/optimization/reprojection_error_only_deformation.h
        ${root}/optimization/reprojection_error_only_pose.h
        ${root}/optimization/reprojection_error_with_deformation.h
        ${root}/optimization/spatial_regularizer.h
        ${root}/optimization/spatial_regularizer_fixed.h
        ${root}/optimization/spatial_regularizer_with_deformation.h
        ${root}/SLAM/settings.h
        ${root}/SLAM/system.h
        ${root}/stereo/stereo_lucas_kanade.h
        ${root}/stereo/stereo_matcher.h
        ${root}/stereo/stereo_pattern_matching.h
        ${root}/tracking/essential_matrix_initialization.h
        ${root}/tracking/monocular_map_initializer.h
        ${root}/tracking/tracking.h
        ${root}/utilities/dbscan.h
        ${root}/utilities/frame_evaluator.h
        ${root}/utilities/geometry_toolbox.h
        ${root}/utilities/landmark_status.h
        ${root}/utilities/statistics_toolbox.h
        ${root}/utilities/time_profiler.h
        ${root}/utilities/types_conversions.h
        ${root}/visualization/color_factory.h
        ${root}/visualization/image_visualizer.h
        ${root}/visualization/map_visualizer.h
        )

set(SOURCE_FILES
        ${root}/calibration/kannala_brandt_8.cc
        ${root}/calibration/pin_hole.cc
        ${root}/datasets/endomapper.cc
        ${root}/datasets/hamlyn.cc
        ${root}/datasets/simulation.cc
        ${root}/features/shi_tomasi.cc
        ${root}/map/frame.cc
        ${root}/map/keyframe.cc
        ${root}/map/map.cc
        ${root}/map/mappoint.cc
        ${root}/map/regularization_graph.cc
        ${root}/map/temporal_buffer.cc
        ${root}/mapping/mapping.cc
        ${root}/masking/bright_filter.cc
        ${root}/masking/border_filter.cc
        ${root}/masking/masker.cc
        ${root}/masking/predefined_filter.cc
        ${root}/matching/lucas_kanade_tracker.cc
        ${root}/optimization/g2o_optimization.cc
        ${root}/optimization/landmark_vertex.cc
        ${root}/optimization/position_regularizer.cc
        ${root}/optimization/position_regularizer_with_deformation.cc
        ${root}/optimization/reprojection_error.cc
        ${root}/optimization/reprojection_error_only_deformation.cc
        ${root}/optimization/reprojection_error_only_pose.cc
        ${root}/optimization/reprojection_error_with_deformation.cc
        ${root}/optimization/spatial_regularizer.cc
        ${root}/optimization/spatial_regularizer_fixed.cc
        ${root}/optimization/spatial_regularizer_with_deformation.cc
        ${root}/optimization/spatial_regularizer_with_observation.cc
        ${root}/SLAM/settings.cc
        ${root}/SLAM/system.cc
        ${root}/stereo/stereo_lucas_kanade.cc
        ${root}/stereo/stereo_pattern_matching.cc
        ${root}/tracking/essential_matrix_initialization.cc
        ${root}/tracking/monocular_map_initializer.cc
        ${root}/tracking/tracking.cc
        ${root}/utilities/dbscan.cc
        ${root}/utilities/frame_evaluator.cc
        ${root}/utilities/geometry_toolbox.cc
        ${root}/utilities/landmark_status.cc
        ${root}/utilities/statistics_toolbox.cc
        ${root}/utilities/time_profiler.cc
        ${root}/utilities/types_conversions.cc
        ${root}/visualization/color_factory.cc
        ${root}/visualization/image_visualizer.cc
        ${root}/visualization/map_visualizer.cc
        )

set(SLAM_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Define the include directory search file for this project
include_directories(${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/third_party/Sophus
        ${CMAKE_SOURCE_DIR}/third_party/g2o
        ${Pangolin_INCLUDE_DIRS}
        ${MLPACK_INCLUDE_DIRS}
        )

#Add the library
add_library(${LIBRARY_NAME} SHARED
        ${SOURCE_FILES}
        ${HEADER_FILES})

# Define the link libraries used in the project
target_link_libraries(${LIBRARY_NAME}
        absl::check
        absl::log
        absl::status
        absl::statusor
        absl::die_if_null
        absl::log_flags
        absl::log_globals
        absl::log_initialize
        absl::log_entry
        absl::log_sink
        absl::log_sink_registry
        absl::log_streamer
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        fmt
        ${Pangolin_LIBRARIES}
        ${MLPACK_LIBRARIES}
        armadillo
        ${CMAKE_SOURCE_DIR}/third_party/g2o/lib/libg2o_core.so
        ${CMAKE_SOURCE_DIR}/third_party/g2o/lib/libg2o_stuff.so
        ${CMAKE_SOURCE_DIR}/third_party/g2o/lib/libg2o_types_sba.so
        )

target_include_directories(NR-SLAM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})