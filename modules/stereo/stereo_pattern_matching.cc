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

#include "stereo_pattern_matching.h"

#include "absl/log/log.h"

#define fx calibration_->GetParameter(0)
#define fy calibration_->GetParameter(1)
#define cx calibration_->GetParameter(2)
#define cy calibration_->GetParameter(3)

StereoPatternMatching::StereoPatternMatching(std::shared_ptr<CameraModel> calibration, float baseline) :
        StereoMatcher(calibration, baseline) {
}

absl::StatusOr<Eigen::Vector3f> StereoPatternMatching::computeStereo3D(const cv::KeyPoint &kp, const cv::Mat &imLeft, const cv::Mat &imRight){
    if (kp.pt.x < 0 || kp.pt.y < 0 || kp.pt.y > imRight.rows - 20 || kp.pt.x > imRight.cols - 20) {
        return absl::InternalError("Feature out of image boundaries");
    }

    int tempx(15), tempy(15);
    int searchx(300), searchy(tempy + 2);

    if (((kp.pt.x - tempx / 2) < 20) or ((kp.pt.y - tempy / 2) < 0) or
        ((kp.pt.x + tempx / 2) > imRight.cols) or
        ((kp.pt.y + tempy / 2) > imRight.rows))
        return absl::InternalError("Feature out of image boundaries");

    cv::Rect cropRect(kp.pt.x - tempx / 2, kp.pt.y - tempy / 2, tempx, tempy);

    int finx = (kp.pt.x - searchx);
    int finy = (kp.pt.y - searchy);
    int finxright = kp.pt.x + baseline_ / 4;
    int finyright = finy + searchy * 2;
    finx = 0;
    finy = 0;
    searchx = float(imRight.cols - 1 - finx) - 2;
    searchy = float(imRight.rows - 1 - finy) / 2 - 2;

    cv::Rect SearchRegion(finx, finy, searchx, searchy * 2);
    cv::Mat tmp = imLeft(cropRect);
    double minValcrop;
    double maxValcrop;
    cv::Point minLoccrop;
    cv::Point maxLoccrop;
    cv::minMaxLoc(tmp, &minValcrop, &maxValcrop, &minLoccrop, &maxLoccrop, cv::Mat());
    if (maxValcrop > 250)
        return absl::InternalError("Feature out of image boundaries");
    cv::Mat SearchImage = imRight(SearchRegion);
    cv::Mat result;
    cv::matchTemplate(SearchImage, tmp, result, cv::TM_CCORR_NORMED);
    cv::Point matchLoc;

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    double CorrelationThreshold(0.99);

    if (maxVal < CorrelationThreshold)
        return absl::InternalError("Feature out of image boundaries");

    matchLoc = maxLoc;

    matchLoc.x = matchLoc.x + finx + tempx / 2;
    matchLoc.y = matchLoc.y + finy + tempy / 2;

    float disp(std::abs(matchLoc.x - kp.pt.x));

    Eigen::Vector3f ps;
    ps.x() = baseline_ / disp * (((float)kp.pt.x - cx) / fx);
    ps.y() = baseline_ / disp * (((float)kp.pt.y - cy) / fy);
    ps.z() = baseline_ / disp;
    return ps;
}

std::vector<absl::StatusOr<Eigen::Vector3f>>
StereoPatternMatching::ComputeStereo3D(const std::vector<cv::KeyPoint> &kp, const cv::Mat &imLeft,
                                       const cv::Mat &imRight) {
    vector<absl::StatusOr<Eigen::Vector3f>> computed3D(kp.size());
    for (int idx = 0; idx < kp.size(); idx++) {
        computed3D[idx] = computeStereo3D(kp[idx], imLeft, imRight);
    }

    return computed3D;
}
