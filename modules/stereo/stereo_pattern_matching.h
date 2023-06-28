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

#ifndef NRSLAM_STEREO_PATTERN_MATCHING_H
#define NRSLAM_STEREO_PATTERN_MATCHING_H


#include "calibration/camera_model.h"
#include "stereo/stereo_matcher.h"

#include "absl/status/statusor.h"

#include <memory>

using namespace std;

class StereoPatternMatching : public StereoMatcher {
public:
    StereoPatternMatching() = delete;
    StereoPatternMatching(std::shared_ptr<CameraModel> calibration, float baseline);

    absl::StatusOr<Eigen::Vector3f> computeStereo3D(const cv::KeyPoint &keypoints,
                                                    const cv::Mat &im_left,
                                                    const cv::Mat &im_right);

    std::vector<absl::StatusOr<Eigen::Vector3f>> ComputeStereo3D(const std::vector<cv::KeyPoint> &keypoints,
                                                                 const cv::Mat &im_left,
                                                                 const cv::Mat &im_right);
private:
};

#endif //NRSLAM_STEREO_PATTERN_MATCHING_H
