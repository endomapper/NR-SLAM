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

#ifndef NRSLAM_HAMLYN_H
#define NRSLAM_HAMLYN_H

#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"

#include <opencv2/opencv.hpp>

class Hamlyn {
public:
    /*
     * Loads the dataset stored at path. If the video has not beeen previously splite, it splits it. Otherwise just loads
     * the images
     */
    Hamlyn(const std::string& video_path, const std::string& other_video_path = "");

    /*
     * Retrieves the i-th image in the sequence
     */
    absl::StatusOr<cv::Mat> GetImage(const int idx);

    absl::StatusOr<cv::Mat> GetRightImage(const int idx);

private:

    /*
     * Splits a given video found at videoPath and splits it into single images stored at path
     */
    bool SplitVideoIntoFrames(const std::string& path, const std::string& video_path,
                              const std::string& other_video_path = "");

    std::vector<std::string> left_images_names_, right_images_names; // Vector with the image paths.

    cv::Mat extrinsicCal_;         // extrinsic calibration.
    cv::Mat leftCal_;              // left Calibration.
    cv::Mat rightCal_;             // right Calibration.
    cv::Mat leftDistorsion_;       // left Distorsion.
    cv::Mat rightDistorsion_;      // right Distorsion.
    cv::Mat R;                     // Rotation between the first and the second camera.
    cv::Mat t;                     // Translation between the first and the second camera.
    cv::Mat R_l, R_r, P_l, P_r, Q; // Rectification parameters.
    cv::Mat M1l, M2l, M1r, M2r;    // Map Parameters.

};

#endif //NRSLAM_HAMLYN_H
