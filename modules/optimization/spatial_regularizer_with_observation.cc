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

#include "spatial_regularizer_with_observation.h"

SpatialRegularizerWithObservation::SpatialRegularizerWithObservation() {};

bool SpatialRegularizerWithObservation::read(std::istream& is){
    return true;
};

bool SpatialRegularizerWithObservation::write(std::ostream& os) const {
    return true;
};

void SpatialRegularizerWithObservation::computeError() {
    const LandmarkVertex* current_landmark_vertex = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* next_landmark_vertex = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector3d current_landmark_position = current_world_transform_camera_ *
            current_landmark_vertex->estimate();
    Eigen::Vector3d next_landmark_position = next_world_transform_camera_ *
            next_landmark_vertex->estimate();

    Eigen::Vector3d obs(_measurement);

    _error = weight_ * (obs - (next_landmark_position - current_landmark_position));
}

 void SpatialRegularizerWithObservation::linearizeOplus() {
    _jacobianOplusXi = weight_ * Eigen::Matrix3d::Identity();
    _jacobianOplusXj = -weight_ * Eigen::Matrix3d::Identity();
}
