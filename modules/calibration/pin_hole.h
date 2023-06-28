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

#ifndef NRSLAM_PIN_HOLE_H
#define NRSLAM_KANNALA_BRANDT_8_H

#include "camera_model.h"

#include <assert.h>
#include <vector>

#include <opencv2/opencv.hpp>

class PinHole : public CameraModel{
public:
    PinHole() {
        calibration_parameters_.resize(4);
    }

    PinHole(const std::vector<float> calibration_parameters) :
    CameraModel(calibration_parameters) {
        assert(calibration_parameters_.size() == 4);
    }

    void Project(const Eigen::Vector3f& landmark_position,
                 Eigen::Vector2f& pixel_position);

    void Unproject(const Eigen::Vector2f& pixel_position,
                   Eigen::Vector3f& projecting_ray);

    void ProjectionJacobian(const Eigen::Vector3f& landmark_position,
                            Eigen::Matrix<float,2,3>& projection_jacobian);

    void UnprojectionJacobian(const Eigen::Vector2f& pixel_position,
                              Eigen::Matrix<float,3,2>& unprojection_jacobian);

    Eigen::Matrix3f ToIntrinsicsMatrix();
};

#endif //NRSLAM_KANNALA_BRANDT_8_H
