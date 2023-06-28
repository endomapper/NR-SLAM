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

#ifndef NRSLAM_STEREO_MATCHER_H
#define NRSLAM_STEREO_MATCHER_H

#include "calibration/camera_model.h"

#include "absl/status/statusor.h"

class StereoMatcher {
public:
    StereoMatcher(std::shared_ptr<CameraModel> calibration, float baseline) :
        calibration_(calibration), baseline_(baseline) {};

    virtual std::vector<absl::StatusOr<Eigen::Vector3f>> ComputeStereo3D(
            const std::vector<cv::KeyPoint>& keypoints,
            const cv::Mat& im_left,
            const cv::Mat& im_right) = 0;

protected:
    std::shared_ptr<CameraModel> calibration_;

    float baseline_;
};


#endif //NRSLAM_STEREO_MATCHER_H
