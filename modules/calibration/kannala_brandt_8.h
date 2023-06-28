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

#ifndef NRSLAM_KANNALA_BRANDT_8_H
#define NRSLAM_KANNALA_BRANDT_8_H

#include "calibration/camera_model.h"

#include <opencv2/opencv.hpp>

class KannalaBrandt8 : public CameraModel {
public:
    KannalaBrandt8() :  precision_(1e-6) {
        calibration_parameters_.resize(8);
    }

    KannalaBrandt8(const std::vector<float> calibration_parameters) :
    CameraModel(calibration_parameters), precision_(1e-6) {
        assert(calibration_parameters_.size() == 8);
    }

    void Project(const Eigen::Vector3f& landmark_position,
                 Eigen::Vector2f& pixel_position);

    void Unproject(const Eigen::Vector2f& p2D,
                   Eigen::Vector3f& projecting_ray);

    void ProjectionJacobian(const Eigen::Vector3f& landmark_position,
                            Eigen::Matrix<float,2,3>& projection_jacobian);

    void UnprojectionJacobian(const Eigen::Vector2f& pixel_position,
                              Eigen::Matrix<float,3,2>& unprojection_jacobian);

    Eigen::Matrix3f ToIntrinsicsMatrix();

private:
    const float precision_;

    // Parameter vector corresponds to:
    // [fx, fy, cx, cy, k0, k1, k2, k3]
};

#endif //NRSLAM_KANNALA_BRANDT_8_H
