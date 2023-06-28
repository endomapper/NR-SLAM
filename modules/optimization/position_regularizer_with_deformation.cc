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
#include "position_regularizer_with_deformation.h"

PositionRegularizerWithDeformation::PositionRegularizerWithDeformation() {}

bool PositionRegularizerWithDeformation::read(std::istream& is) {
    return true;
}

bool PositionRegularizerWithDeformation::write(std::ostream& os) const {
    return true;
}

void PositionRegularizerWithDeformation::computeError() {
    const LandmarkVertex* vertex_flow_1 = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* vertex_flow_2 = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector3d current_position_1 = rest_position_1_ + vertex_flow_1->estimate();
    Eigen::Vector3d current_position_2 = rest_position_2_ + vertex_flow_2->estimate();

    double current_distance = (current_position_1 - current_position_2).norm();

    _error(0) = k_ * (current_distance - _measurement) / _measurement;
}

void PositionRegularizerWithDeformation::linearizeOplus() {
    const LandmarkVertex* vertex_flow_1 = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* vertex_flow_2 = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector3d current_position_1 = rest_position_1_ + vertex_flow_1->estimate();
    Eigen::Vector3d current_position_2 = rest_position_2_ + vertex_flow_2->estimate();

    double current_distance = (current_position_1 - current_position_2).norm();

    double a = k_ / (2 * _measurement * current_distance);
    Eigen::Vector3d v = 2 * current_position_1 - 2 * current_position_2;

    _jacobianOplusXi = a * v.transpose();
    _jacobianOplusXj = -a * v.transpose();
}
