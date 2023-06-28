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

#ifndef NRSLAM_GEOMETRY_TOOLBOX_H
#define NRSLAM_GEOMETRY_TOOLBOX_H

#include "absl/status/statusor.h"

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include "sophus/se3.hpp"

float InterpolationWeight(const float distance, const float sigma);

float SquaredReprojectionError(cv::Point2f point_1, cv::Point2f point2);

// Computes the cosine of the parallax angle between 2 rays.
float RaysParallaxCosine(const Eigen::Vector3f& ray_1, const Eigen::Vector3f& ray_2);

// Computes the parallax angle (in rads) between 2 rays
float RaysParallax(const Eigen::Vector3f& ray_1, const Eigen::Vector3f& ray_2);

// Triangulates a 3D point form the camera poses and the normalized bearing rays.
// Lee, Seong Hun, and Javier Civera. "Triangulation: Why Optimize?." arXiv preprint arXiv:1907.11917 (2019).
absl::StatusOr<Eigen::Vector3f> TriangulateMidPoint(const Eigen::Vector3f &ray_1, const Eigen::Vector3f &ray_2,
                         const Sophus::SE3f &camera1_transform_world, const Sophus::SE3f &camera2_transform_world);


// Interpolates a value in a matrix given a floating point position.
template<typename T>
float Interpolate(const float x, const float y, const T* mat, const int cols){
    float x_,_x,y_,_y;
    _x = modf(x,&x_);
    _y = modf(y,&y_);

    //Get interpolation weights
    float w00 = (1.f - _x)*(1.f - _y);
    float w01 = (1.f - _x)*_y;
    float w10 = _x*(1.f -_y);
    float w11 = 1.f - w00 - w01 - w10;

    return (float)(mat[(int)y_*cols+(int)x_])*w00 + (float)(mat[(int)y_*cols+(int)x_+1])*w10 +
           (float)(mat[((int)y_+1)*cols+(int)x_])*w01 + (float)(mat[((int)y_+1)*cols+(int)x_+1])*w11;
}

template<typename Derived>
inline bool HasInf(const Eigen::MatrixBase<Derived>& x){
    float sum = x.sum();
    if(isinf(sum) || isinf(-sum) || isnan(sum)){
        return true;
    }
    else{
        return false;
    }
}

#endif //NRSLAM_GEOMETRY_TOOLBOX_H
