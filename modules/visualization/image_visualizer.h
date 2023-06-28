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

#ifndef NRSLAM_IMAGE_VISUALIZER_H
#define NRSLAM_IMAGE_VISUALIZER_H

#include "map/frame.h"
#include "map/regularization_graph.h"
#include "map/temporal_buffer.h"
#include "color_factory.h"

#include <opencv2/opencv.hpp>

class ImageVisualizer {
public:
    struct Options {
        bool wait_for_user_button = false;

        std::string image_save_path;
    };

    ImageVisualizer() = delete;

    ImageVisualizer(Options& options);

    void SetCurrentImage(const cv::Mat& im_original, const cv::Mat& im_processed);

    void DrawCurrentFrame(Frame& frame, const bool use_original_image = true);

    void DrawFrame(Frame& frame, std::string name);

    void DrawRegularizationGraph(Frame& frame, RegularizationGraph& regularization_graph,
                                 const bool use_original_image = true);

    void DrawOpticalFlow(TemporalBuffer& temporal_buffer);

    void DrawClusteredOpticalFlow(std::vector<std::vector<cv::Point2f>>& feature_tracks,
                                  std::vector<int> point_labels,
                                  const bool use_original_image = true);

    void DrawFeatures(std::vector<cv::KeyPoint>& keypoints,
                      const bool use_original_image = true);

    void DrawFeatures(std::vector<cv::KeyPoint>& keypoints,
                      std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
                      const bool use_original_image = true);

    void UpdateWindows();

    int GetCurrentImageNumber();

private:
    void DrawFeature(cv::Mat& im, cv::Point2f uv, cv::Scalar color, int size = 5);

    cv::Scalar HeatMapColor(const float min_value, const float max_value, const float value);

    cv::Scalar OpenCVHeatMapColor(const float min_value, const float max_value, const float value,
                                  const int colormap);

    cv::Mat current_color_image_, current_processed_image_;

    ColorFactory color_factory_;

    int current_image_number_;

    Options options_;
};


#endif //NRSLAM_IMAGE_VISUALIZER_H
