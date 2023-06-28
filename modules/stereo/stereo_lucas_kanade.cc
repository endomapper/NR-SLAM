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

#include "stereo_lucas_kanade.h"
#include "absl/log/check.h"

using namespace std;

#define fx calibration_->GetParameter(0)
#define fy calibration_->GetParameter(1)
#define cx calibration_->GetParameter(2)
#define cy calibration_->GetParameter(3)

StereoLucasKanade::StereoLucasKanade(StereoLucasKanade::Options &options,
                                     std::shared_ptr<CameraModel> calibration, float baseline) :
        options_(options), StereoMatcher(calibration, baseline) {
    klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                      options_.klt_max_level, options_.klt_max_iters,
                                      options_.klt_epsilon, options_.klt_min_eig_th);
}

std::vector<absl::StatusOr<Eigen::Vector3f>>
StereoLucasKanade::ComputeStereo3D(const std::vector<cv::KeyPoint> &keypoints,
                                   const cv::Mat &im_left, const cv::Mat &im_right) {
    klt_tracker_.SetReferenceImage(im_left, keypoints);

    vector<cv::KeyPoint> keypoints_right = keypoints;
    vector<LandmarkStatus> landmark_statuses(keypoints.size(), TRACKED);

    klt_tracker_.Track(im_right, keypoints_right, landmark_statuses, true,
                       options_.klt_min_SSIM, cv::Mat());

    vector<absl::StatusOr<Eigen::Vector3f>> computed3D;
    for (int idx = 0; idx < keypoints.size(); idx++) {
        if (landmark_statuses[idx] == TRACKED) {
            float dif_in_rows = abs(keypoints[idx].pt.y - keypoints_right[idx].pt.y);

            if(dif_in_rows > 2.0) {
                computed3D.push_back(absl::InternalError("Rows are different: " + to_string(dif_in_rows)));
            } else {
                float disparity = (abs(keypoints[idx].pt.x - keypoints_right[idx].pt.x));

                CHECK(!isnan(disparity) && !isinf(disparity));

                Eigen::Vector3f landmark_position;
                landmark_position.x() = baseline_ / disparity * (((float)keypoints[idx].pt.x - cx) / fx);
                landmark_position.y() = baseline_ / disparity * (((float)keypoints[idx].pt.y - cy) / fy);
                landmark_position.z() = baseline_ / disparity;

                computed3D.push_back(landmark_position);
            }

        } else {
            computed3D.push_back(absl::InternalError("KLT did not find a match."));
        }
    }

    return computed3D;
}
