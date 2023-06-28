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

#include "types_conversions.h"

#include <eigen3/Eigen/Geometry>


Sophus::SE3<float> cvToSophus(const cv::Mat &T) {
    Eigen::Matrix<float,3,3> eigMat = cvToEigenM3f(T.rowRange(0,3).colRange(0,3));
    Eigen::Quaternionf q(eigMat.cast<float>());

    Eigen::Matrix<float,3,1> t = cvToEigenV3f(T.rowRange(0,3).col(3)).cast<float>();

    return Sophus::SE3<float>(q,t);
}

Eigen::Matrix<float,3,3> cvToEigenM3f(const cv::Mat &cvMat3) {
    Eigen::Matrix<float,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
            cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
            cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Eigen::Matrix<float,3,1> cvToEigenV3f(const cv::Mat &cvVector) {
    Eigen::Matrix<float,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}