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

#include "frame_evaluator.h"

#include <fstream>

#include "absl/log/log.h"
#include "geometry_toolbox.h"
#include "statistics_toolbox.h"
#include "absl/log/check.h"

using namespace std;

FrameEvaluator::FrameEvaluator(Options& options, std::shared_ptr<StereoMatcher> stereo_matcher,
                               MapVisualizer* map_visualizer) :
        options_(options), stereo_matcher_(stereo_matcher), map_visualizer_(map_visualizer) {}

void FrameEvaluator::EvaluateFrameReconstruction(Frame &frame, const cv::Mat &im_left,
                                                 const cv::Mat &im_right) {
    auto keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    auto landmark_positions = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    auto indices_in_frame = frame.GetIndexWithStatus({TRACKED_WITH_3D});

    auto transformed_landmark_positions = TransformPointCloud(landmark_positions,
                                                              frame.CameraTransformationWorld());

    auto landmarks_ground_truth = ComputeGroundTruth(keypoints, im_left, im_right, frame.GetCalibration());

    auto [computed_error, scale_factor] = ComputeReconstructionRMSE(transformed_landmark_positions, landmarks_ground_truth,
                                                     true);

    SaveGroundTruthToFrame(frame, scale_factor, landmarks_ground_truth, indices_in_frame);

    computed_rmse_.push_back(computed_error);
}

std::tuple<float, float> FrameEvaluator::ComputeReconstructionRMSE(const std::vector<Eigen::Vector3f> &reconstruction_landmarks,
                                                const std::vector<absl::StatusOr<Eigen::Vector3f>> &ground_truth_landmarks,
                                                const bool align_scales) {
    vector<float> estimated_depths, ground_truth_depths;

    for (int idx = 0; idx < reconstruction_landmarks.size(); idx++) {
        if(ground_truth_landmarks[idx].ok()) {
            estimated_depths.push_back(reconstruction_landmarks[idx].z());
            ground_truth_depths.push_back((*ground_truth_landmarks[idx]).z());
        }
    }

    if (align_scales) {
        return ComputeRMSEWithScaleAlignment(estimated_depths, ground_truth_depths);
    } else {
        return ComputeRMSEWithoutScaleAlignment(estimated_depths, ground_truth_depths);
    }
}

std::tuple<float, float> FrameEvaluator::ComputeRMSEWithoutScaleAlignment(const std::vector<float> &estimated_depths,
                                                       const std::vector<float> &ground_truth_depths) {
    int n_depths = estimated_depths.size();

    vector<float> errors;
    vector<float> estimated_inlier_depths, ground_truth_inlier_depths;
    for (int idx = 0; idx < n_depths; idx++) {
        errors.push_back(fabs(estimated_depths[idx] - ground_truth_depths[idx]));
    }

    // Compute inter-quartile range
    vector<float> sorted_errors = errors;
    sort(sorted_errors.begin(), sorted_errors.end());
    float interquartileRange = sorted_errors[(int)(sorted_errors.size() * 0.75f)] - sorted_errors[(int)(sorted_errors.size() * 0.25f)];
    float q1 = sorted_errors[(int)(sorted_errors.size() * 0.25f)];
    float q3 = sorted_errors[(int)(sorted_errors.size() * 0.75f)];

    float th_ = 1.5f * interquartileRange;

    for(int idx = 0; idx < errors.size(); idx++){
        float error = errors[idx];

        if(error <= q3 + th_){
            estimated_inlier_depths.push_back(estimated_depths[idx]);
            ground_truth_inlier_depths.push_back(ground_truth_depths[idx]);
        }
    }

    n_depths = estimated_inlier_depths.size();

    float inlier_fraction = 0.9;
    int n_inliers = n_depths * inlier_fraction;

    Eigen::VectorXf estimated_depths_eigen(n_depths);
    Eigen::VectorXf ground_truth_depths_eigen(n_depths);

    for (int idx = 0; idx < n_depths; idx++){
        estimated_depths_eigen(idx) = estimated_inlier_depths[idx];
        ground_truth_depths_eigen(idx) = ground_truth_inlier_depths[idx];
    }

    Eigen::VectorXf residuals = ground_truth_depths_eigen - estimated_depths_eigen;
    Eigen::VectorXf squared_residuals = (residuals.array() * residuals.array()).matrix();
    Eigen::VectorXf sorted_residuals = squared_residuals;

    sort(sorted_residuals.data(), sorted_residuals.data() + n_depths);
    float th = sorted_residuals(n_inliers);

    // Get only inlier data.
    Eigen::VectorXf inlier_residuals(n_inliers);
    int current_idx = 0;
    for (int idx = 0; idx < n_depths; idx++) {
        if (squared_residuals(idx) < th) {
            inlier_residuals(current_idx) = residuals(idx);
            current_idx++;
        }
    }

    return make_tuple(sqrt(inlier_residuals.dot(inlier_residuals) / (float) n_inliers), 1.f);
}

std::tuple<float, float> FrameEvaluator::ComputeRMSEWithScaleAlignment(const std::vector<float> &estimated_depths,
                                                    const std::vector<float> &ground_truth_depths) {
    int n_depths = estimated_depths.size();

    vector<float> errors;
    vector<float> estimated_inlier_depths, ground_truth_inlier_depths;
    for (int idx = 0; idx < n_depths; idx++) {
        errors.push_back(fabs(estimated_depths[idx] - ground_truth_depths[idx]));
    }

    // Compute inter-quartile range
    vector<float> sorted_errors = errors;
    sort(sorted_errors.begin(), sorted_errors.end());
    float interquartileRange = sorted_errors[(int)(sorted_errors.size() * 0.75f)] - sorted_errors[(int)(sorted_errors.size() * 0.25f)];
    float q1 = sorted_errors[(int)(sorted_errors.size() * 0.25f)];
    float q3 = sorted_errors[(int)(sorted_errors.size() * 0.75f)];

    float th_ = 1.5f * interquartileRange;

    for(int idx = 0; idx < errors.size(); idx++){
        float error = errors[idx];

        if(options_.precomputed_depth_ || error <= q3 + th_){
            estimated_inlier_depths.push_back(estimated_depths[idx]);
            ground_truth_inlier_depths.push_back(ground_truth_depths[idx]);
        }
    }

    n_depths = estimated_inlier_depths.size();
    const float inlier_fraction = (options_.precomputed_depth_) ? 0.95 : 0.9;
    const int n_inliers = (n_depths * inlier_fraction);

    Eigen::VectorXf estimated_depths_eigen(n_depths);
    Eigen::VectorXf ground_truth_depths_eigen(n_depths);

    for (int idx = 0; idx < n_depths; idx++){
        estimated_depths_eigen(idx) = estimated_inlier_depths[idx];
        ground_truth_depths_eigen(idx) = ground_truth_inlier_depths[idx];
    }

    CHECK(!estimated_depths_eigen.hasNaN());
    CHECK(!ground_truth_depths_eigen.hasNaN());

    CHECK(!HasInf(estimated_depths_eigen));
    CHECK(!HasInf(ground_truth_depths_eigen));

    float scale_factor = 1.0f;
    float rmse;

    for(int it = 0; it < 10; it++) {
        // Get all errors.
        Eigen::VectorXf squared_residuals = ((ground_truth_depths_eigen - scale_factor * estimated_depths_eigen).array() *
                (ground_truth_depths_eigen - scale_factor * estimated_depths_eigen).array()).matrix();
        Eigen::VectorXf squared_residuals_sorted = squared_residuals;

        // Get threshold to discard outliers.
        sort(squared_residuals_sorted.data(), squared_residuals_sorted.data() + n_depths);
        float th = squared_residuals_sorted(n_inliers - 1);

        // Get only inlier data.
        Eigen::VectorXf inlier_depths(n_inliers);
        Eigen::VectorXf inlier_groundtruth_depths(n_inliers);
        Eigen::VectorXf inlier_residuals(n_inliers);
        Eigen::VectorXf inlier_jacobians(n_inliers);

        Eigen::VectorXf residuals = ground_truth_depths_eigen - scale_factor * estimated_depths_eigen;

        int current_idx = 0;
        for(int idx = 0; idx < n_depths; idx++){
            if(squared_residuals(idx) <= th) {
                inlier_depths(current_idx) = estimated_depths_eigen(idx);
                inlier_groundtruth_depths(current_idx) = ground_truth_depths_eigen(idx);
                inlier_residuals(current_idx) = residuals(idx);
                inlier_jacobians(current_idx) = -residuals(idx) * estimated_depths_eigen(idx);

                current_idx++;
            }
        }

        const float H = inlier_depths.transpose() * inlier_depths;
        const float g = inlier_jacobians.sum();
        const float delta = -g / H;

        scale_factor += delta;

        Eigen::VectorXf aligned_residuals = inlier_groundtruth_depths - scale_factor * inlier_depths;
        rmse = sqrt(aligned_residuals.dot(aligned_residuals) / (float) n_inliers);
    }

    CHECK(!isinf(rmse));

    return make_tuple(rmse, scale_factor);
}

std::vector<Eigen::Vector3f> FrameEvaluator::TransformPointCloud(const std::vector<Eigen::Vector3f> &point_cloud,
                                                                 const Sophus::SE3f &camera_transform_world) {
    vector<Eigen::Vector3f> transformed_point_cloud;
    for (auto point : point_cloud) {
        transformed_point_cloud.push_back(camera_transform_world * point);
    }

    return transformed_point_cloud;
}

std::vector<absl::StatusOr<Eigen::Vector3f>>
FrameEvaluator::ComputeGroundTruth(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &im_left,
                                        const cv::Mat &im_right, std::shared_ptr<CameraModel> calibration) {
    if (!options_.precomputed_depth_) {
        auto raw_ground_truth = stereo_matcher_->ComputeStereo3D(keypoints, im_left, im_right);

        for (int idx = 0; idx < raw_ground_truth.size(); idx++) {
            if(raw_ground_truth[idx].ok()) {
                // These checks are used to remove outliers in sequence when
                // the depth bounds are clear.
                /* if ((*raw_ground_truth[idx]).z() < 40.f ||
                    (*raw_ground_truth[idx]).z() > 100.f) {
                    raw_ground_truth[idx] = absl::InternalError("Wrong depth.");

                    continue;
                } */

                /* if ((*raw_ground_truth[idx]).z() < 100.f ||
                        (*raw_ground_truth[idx]).z() > 175.f){
                    raw_ground_truth[idx] = absl::InternalError("Wrong depth.");

                    continue;
                } */
            }
        }

        return raw_ground_truth;
    } else {
        std::vector<absl::StatusOr<Eigen::Vector3f>> raw_ground_truth;
        for (auto keypoint : keypoints) {
            float ground_truth_depth = Interpolate(keypoint.pt.x, keypoint.pt.y,
                                                   im_right.ptr<float>(0), im_right.cols);

            Eigen::Vector3f projecting_ray = calibration->Unproject(keypoint.pt);
            projecting_ray /= projecting_ray.z();

            raw_ground_truth.push_back(projecting_ray * ground_truth_depth);
        }

        return raw_ground_truth;
    }
}

void FrameEvaluator::SaveResultsToFile() {
    ofstream results_file(options_.results_file_path);

    for (auto error : computed_rmse_) {
        results_file << error << endl;
    }

    results_file.close();
}

void FrameEvaluator::SaveGroundTruthToFrame(Frame &frame, const float scale_factor,
                                            const std::vector<absl::StatusOr<Eigen::Vector3f>> &ground_truth,
                                            const std::vector<int> indices_in_frame) {
    std::vector<absl::StatusOr<Eigen::Vector3f>>& frame_ground_truth = frame.MutableGroundTruth();
    fill(frame_ground_truth.begin(), frame_ground_truth.end(),
         absl::InternalError("No ground truth available."));

    Sophus::SE3f world_transform_camera = frame.CameraTransformationWorld().inverse();

    for (int idx = 0; idx < ground_truth.size(); idx++) {
        if (ground_truth[idx].ok()) {
            const int frame_index = indices_in_frame[idx];
            frame_ground_truth[frame_index] = world_transform_camera * (*ground_truth[idx] / scale_factor);
        }
    }
}
