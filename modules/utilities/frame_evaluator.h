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

#ifndef NRSLAM_FRAME_EVALUATOR_H
#define NRSLAM_FRAME_EVALUATOR_H

#include "map/frame.h"
#include "stereo/stereo_lucas_kanade.h"
#include "stereo/stereo_matcher.h"
#include "visualization/map_visualizer.h"

class FrameEvaluator {
public:
    struct Options {
        std::string results_file_path;
        bool precomputed_depth_;
    };

    FrameEvaluator() = delete;

    FrameEvaluator(Options& options, std::shared_ptr<StereoMatcher> stereo_matcher,
                   MapVisualizer* map_visualizer);

    void EvaluateFrameReconstruction(Frame& frame, const cv::Mat& im_left, const cv::Mat& im_right);

    void SaveResultsToFile();
private:
    std::vector<absl::StatusOr<Eigen::Vector3f>> ComputeGroundTruth(const std::vector<cv::KeyPoint>& keypoints,
                                               const cv::Mat& im_left, const cv::Mat& im_right,
                                               std::shared_ptr<CameraModel> calibration);

    std::tuple<float, float> ComputeReconstructionRMSE(const std::vector<Eigen::Vector3f> &reconstruction_landmarks,
                                    const std::vector<absl::StatusOr<Eigen::Vector3f>> &ground_truth_landmarks,
                                    const bool align_scales);

    std::vector<Eigen::Vector3f> TransformPointCloud(const std::vector<Eigen::Vector3f>& point_cloud,
                                                     const Sophus::SE3f& camera_transform_world);

    std::tuple<float, float> ComputeRMSEWithoutScaleAlignment(const std::vector<float>& estimated_depths,
                                           const std::vector<float>& ground_truth_depths);

    std::tuple<float, float> ComputeRMSEWithScaleAlignment(const std::vector<float>& estimated_depths,
                                           const std::vector<float>& ground_truth_depths);

    void SaveGroundTruthToFrame(Frame& frame, const float scale_factor,
                                const std::vector<absl::StatusOr<Eigen::Vector3f>>& ground_truth,
                                const std::vector<int> indices_in_frame);

    Options options_;

    std::shared_ptr<StereoMatcher> stereo_matcher_;

    MapVisualizer* map_visualizer_;

    std::vector<float> computed_rmse_;
};


#endif //NRSLAM_FRAME_EVALUATOR_H
