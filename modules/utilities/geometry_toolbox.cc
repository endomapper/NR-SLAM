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

#include "geometry_toolbox.h"

#include <cmath>

using namespace std;

float InterpolationWeight(const float distance, const float sigma) {
    return exp(-(distance * distance) / (2 * sigma * sigma));
}

float SquaredReprojectionError(cv::Point2f point_1, cv::Point2f point2) {
    float errx = point_1.x - point2.x;
    float erry = point_1.y - point2.y;

    return errx * errx + erry *erry;
}

float RaysParallaxCosine(const Eigen::Vector3f& ray_1, const Eigen::Vector3f& ray_2) {
    return ray_1.dot(ray_2) / (ray_1.norm() * ray_2.norm());
}

float RaysParallax(const Eigen::Vector3f& ray_1, const Eigen::Vector3f& ray_2) {
    return acos(min(RaysParallaxCosine(ray_1, ray_2),1.f));
}

absl::StatusOr<Eigen::Vector3f> TriangulateMidPoint(const Eigen::Vector3f &ray_1, const Eigen::Vector3f &ray_2,
                                                    const Sophus::SE3f &camera1_transform_world, const Sophus::SE3f &camera2_transform_world) {
    // Data definition using the paper variable naming.
    Eigen::Vector3f f0 = ray_1;
    Eigen::Vector3f f1 = ray_2;

    Eigen::Vector3f f0_hat = ray_1.normalized();
    Eigen::Vector3f f1_hat = ray_2.normalized();

    Sophus::SE3f T10 = camera2_transform_world * camera1_transform_world.inverse();
    Eigen::Vector3f t = T10.translation();
    Eigen::Matrix3f R = T10.rotationMatrix();

    // Depth computation.
    Eigen::Vector3f p = (R * f0_hat).cross(f1_hat);
    Eigen::Vector3f q = (R * f0_hat).cross(t);
    Eigen::Vector3f r = f1_hat.cross(t);

    float lambda0 = r.norm() / p.norm();
    float lambda1 = q.norm() / p.norm();

    // Adequacy test.
    Eigen::Vector3f point0 = lambda0 * R * f0_hat;
    Eigen::Vector3f point1 = lambda1 * f1_hat;

    float v1 = (t + point0 + point1).squaredNorm();
    float v2 = (t - point0 - point1).squaredNorm();
    float v3 = (t - point0 + point1).squaredNorm();

    float minv = fmin(v1,fmin(v2,v3));

    // Inverse Depth Weighted MidPoint.
    Eigen::Vector3f x1 = q.norm() / (q.norm() + r.norm()) * (t + r.norm() / p.norm() * (R * f0_hat + f1_hat));
    return camera2_transform_world.inverse() * x1;
}
