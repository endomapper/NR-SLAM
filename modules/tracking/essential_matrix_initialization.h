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

#ifndef NRSLAM_ESSENTIAL_MATRIX_INITIALIZATION_H
#define NRSLAM_ESSENTIAL_MATRIX_INITIALIZATION_H

#include <memory>

#include "calibration/camera_model.h"
#include "utilities/landmark_status.h"
#include "visualization/image_visualizer.h"

#include "absl/status/statusor.h"

#include "sophus/se3.hpp"

class EssentialMatrixInitialization {
public:
    struct Options {
        int max_features;
        int min_sample_set_size;
        float min_parallax;
        float radians_per_pixel;
        float epipolar_threshold;
    };

    EssentialMatrixInitialization() = delete;

    // Constructor with the max number of expected features, the calibration and
    // thresholds for reconstruction.
    EssentialMatrixInitialization(Options& options,
                                  std::shared_ptr<CameraModel> calibration,
                                  std::shared_ptr<ImageVisualizer> image_visualizer = nullptr);

    // Changes the reference features for initialization.
    void ChangeReference(std::vector<cv::KeyPoint>& keypoints);

    // Tries to initialize using the reference and current feature tracks. Returns Ok in
    // success and camera_transform_world and landmarks_position hold valid data.
    absl::Status Initialize(const std::vector<cv::KeyPoint>& current_keypoints,
                    const std::vector<LandmarkStatus>& keypoint_statuses,
                    const int n_matches, Sophus::SE3f& camera_transform_world,
                    std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position);

private:
    // Unprojects tracked features.
    void UnprojectTrackedFeatures();

    // Finds an Essential matrix with RANSAC.
    Eigen::Matrix3f FindEssentialWithRANSAC(const int n_matches,
                                            std::vector<bool>& inliers,
                                            int& n_inliers);

    // Computes an Essential Matrix from a minimum set of bearing rays.
    Eigen::Matrix3f ComputeE(Eigen::Matrix<float,8,3>& reference_rays_sample,
                             Eigen::Matrix<float,8,3>& current_rays_sample);

    Eigen::Matrix3f RefineSolution(Eigen::MatrixXf& reference_rays,
                                   Eigen::MatrixXf& current_rays);

    // Computes the number of inliers of a given Essential matrix with the current data.
    int ComputeScoreAndInliers(const int n_matched,
                               Eigen::Matrix<float,3,3>& E,
                               std::vector<bool>& inliers);

    // Environment reconstruction from a given Essential Matrix.
    absl::Status ReconstructEnvironment(Eigen::Matrix3f& E, Sophus::SE3f& camera_transform_world,
                                std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
                                int& n_inliers, std::vector<bool>& essential_matrix_inliers);

    //Reconstructs the camera pose from the Essential Matrix. Automatically sellects the correct rotaiton and translation
    void ReconstructCameras(Eigen::Matrix3f& E ,Sophus::SE3f& camera_transform_world,
                            Eigen::MatrixXf& rays_1, Eigen::MatrixXf& rays_2);

    //Reconstructs the environment with te predicted camera pose
    absl::Status ReconstructPoints(const Sophus::SE3f& camera_transform_world,
                           std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
                           std::vector<bool>& essential_matrix_inliers);

    //Decompose an Essential matrix into the 2 possible rotations and translations
    void DecomposeEssentialMatrix(Eigen::Matrix3f& E, Eigen::Matrix3f& R_1,
                                  Eigen::Matrix3f& R_2, Eigen::Vector3f& t);

    // Computes the maximum number of iterations for the RANSAC.
    int ComputeMaxTries(const float inlier_fraction, const float success_likelihood);

    Options options_;

    // Reference and current KeyPoints.
    std::vector<cv::KeyPoint> reference_keypoints_, current_keypoints_;
    std::vector<cv::Point2f> reference_keypoints_tracked_;

    std::vector<int> ransac_idx_to_keypoint_idx_;

    // Status of the feature tracks.
    std::vector<LandmarkStatus> feature_tracks_statuses_;

    // Bearing rays (unprojected points) of the reference and current KeyPoints.
    Eigen::Matrix<float,Eigen::Dynamic,3,Eigen::RowMajor> reference_rays_, current_rays_;

    //Calibration of the reference and current view
    std::shared_ptr<CameraModel> calibration_;

    std::shared_ptr<ImageVisualizer> image_visualizer_;
};


#endif //NRSLAM_ESSENTIAL_MATRIX_INITIALIZATION_H
