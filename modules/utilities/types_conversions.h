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

#ifndef NRSLAM_TYPES_CONVERSIONS_H
#define NRSLAM_TYPES_CONVERSIONS_H

#include <sophus/se3.hpp>
#include <opencv2/core/mat.hpp>
#include <eigen3/Eigen/Core>

Sophus::SE3<float> cvToSophus(const cv::Mat& T);

Eigen::Matrix<float,3,3> cvToEigenM3f(const cv::Mat &cvMat3);

Eigen::Matrix<float,3,1> cvToEigenV3f(const cv::Mat &cvVector);


#endif //NRSLAM_TYPES_CONVERSIONS_H
