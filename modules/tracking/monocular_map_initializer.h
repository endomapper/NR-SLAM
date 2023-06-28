/*
 * This file is part of NR-SLAM
 *
 * Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * NR-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NRSLAM_MONOCULAR_MAP_INITIALIZER_H
#define NRSLAM_MONOCULAR_MAP_INITIALIZER_H

#include "features/feature.h"
#include "matching/lucas_kanade_tracker.h"
#include "tracking/essential_matrix_initialization.h"
#include "visualization/image_visualizer.h"

#include "absl/container/flat_hash_map.h"
#include <opencv2/opencv.hpp>

class MonocularMapInitializer {
public:
    struct Options {
        int klt_window_size = 21;
        int klt_max_level = 3;
        int klt_max_iters = 50;
        float klt_epsilon = 0.01;
        float klt_min_eig_th = 1e-4;
        float klt_min_SSIM = 0.7;

        int rigid_initializer_max_features;
        int rigid_initializer_min_sample_set_size;
        float rigid_initializer_min_parallax;
        float rigid_initializer_radians_per_pixel;
        float rigid_initializer_epipolar_threshold;
    };

    struct InitializationResults {
        Sophus::SE3f camera_transform_world;

        std::vector<cv::KeyPoint> reference_keypoints;
        std::vector<cv::KeyPoint> current_keypoints;

        std::vector<Eigen::Vector3f> reference_landmark_positions;
        std::vector<Eigen::Vector3f> current_landmark_positions;
    };

    MonocularMapInitializer() = delete;

    MonocularMapInitializer(Options& options, std::shared_ptr<Feature> feature_extractor,
                            std::shared_ptr<CameraModel> calibration,
                            std::shared_ptr<ImageVisualizer> image_visualizer);

    absl::StatusOr<InitializationResults> ProcessNewImage(const cv::Mat& im, const cv::Mat& im_clahe,
                         const cv::Mat& mask);

private:
    void DataAssociation(const cv::Mat& im, const cv::Mat& im_clahe,
                         const cv::Mat& mask);

    void ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                         std::vector<cv::KeyPoint>& keypoints);

    void AddFeatureTracks(const std::vector<cv::KeyPoint>& keypoints,
                          const std::vector<LandmarkStatus>& keypoint_statuses);

    void UpdateTrackingReference(const cv::Mat& im);

    std::vector<int> FeatureTracksClustering();

    void ResetInitialization(const cv::Mat& im, const cv::Mat& im_clahe,
                             const cv::Mat& mask);

    typedef absl::StatusOr<
            std::tuple<Sophus::SE3f, std::vector<absl::StatusOr<Eigen::Vector3f>>>>
            RigidInitializationResults;

    RigidInitializationResults RigidInitialization();

    InitializationResults InitializationRefinement(std::vector<cv::KeyPoint>& current_keypoints,
                                  std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
                                  std::vector<int>& feature_labels,
                                  Sophus::SE3f& camera_transform_world);

    InitializationResults BuildInitializationResults(std::vector<std::vector<cv::KeyPoint>>& feature_tracks,
                                                std::vector<std::vector<Eigen::Vector3f>>& landmark_tracks,
                                                std::vector<int>& track_labels,
                                                std::vector<Sophus::SE3f>& camera_trajectory);

    enum InternalStatus {
        NO_DATA,
        OK,
        RECENTLY_RESET
    };

    InternalStatus internal_status_;

    struct FeatureTrack {
        std::vector<cv::KeyPoint> track_;
    };

    struct FeatureTracks {
        int max_feature_track_lenght = 0;

        absl::flat_hash_map<int, FeatureTrack> feature_id_to_feature_track;
    };

    Options options_;

    std::shared_ptr<Feature> feature_extractor_;

    LucasKanadeTracker klt_tracker_;

    FeatureTracks feature_tracks_;

    std::vector<cv::KeyPoint> current_keypoints_;

    std::vector<LandmarkStatus> current_keypoint_statuses_;

    int images_from_last_reference_ = 0;

    std::shared_ptr<ImageVisualizer> image_visualizer_;

    std::unique_ptr<EssentialMatrixInitialization> rigid_initializer_;

    int n_tracks_in_image_;

    std::shared_ptr<CameraModel> calibration_;
};


#endif //NRSLAM_MONOCULAR_MAP_INITIALIZER_H
