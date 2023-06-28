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

#ifndef NRSLAM_SIMULATION_H
#define NRSLAM_SIMULATION_H

#include <string>

#include "absl/status/statusor.h"

#include <opencv2/opencv.hpp>
#include "sophus/se3.hpp"

class Simulation {
public:
    Simulation(const std::string& dataset_path);

    absl::StatusOr<cv::Mat> GetImage(const int idx);

    absl::StatusOr<cv::Mat> GetDepthImage(const int idx);

    absl::StatusOr<Sophus::SE3f> GetCameraPose(const int idx);

private:
    void GenerateNamesFile(const std::string& images_path);

    // Vector with the image paths.
    std::vector<std::string> images_names_;

    // Vector with the depth image paths.
    std::vector<std::string> depth_images_names_;
    std::vector<Sophus::SE3f> ground_truth_poses_;

    const float far_clip_ = 4.f;
    const float near_clip_ = 0.01f;
};


#endif //NRSLAM_SIMULATION_H
