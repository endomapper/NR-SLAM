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

#include "spatial_regularizer.h"

SpatialRegularizer::SpatialRegularizer(){};

bool SpatialRegularizer::read(std::istream& is){
    return true;
};

bool SpatialRegularizer::write(std::ostream& os) const{
    return true;
};

void SpatialRegularizer::computeError() {
    const LandmarkVertex* vertex_point_1_current = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* vertex_point_2_current = static_cast<const LandmarkVertex*>(_vertices[1]);

    const LandmarkVertex* vertex_point_1_next = static_cast<const LandmarkVertex*>(_vertices[2]);
    const LandmarkVertex* vertex_point_2_next = static_cast<const LandmarkVertex*>(_vertices[3]);

    Eigen::Vector3d point_1_current = vertex_point_1_current->estimate();
    Eigen::Vector3d point_2_current = vertex_point_2_current->estimate();

    Eigen::Vector3d point_1_next = vertex_point_1_next->estimate();
    Eigen::Vector3d point_2_next = vertex_point_2_next->estimate();

    _error = weight_ * ((point_1_next - point_1_current) -
            (point_2_next - point_2_current));
}

void SpatialRegularizer::linearizeOplus() {
    auto& jacobian_wrt_x_1_current = std::get<0>(this->_jacobianOplus);
    auto& jacobian_wrt_x_2_current = std::get<1>(this->_jacobianOplus);
    auto& jacobian_wrt_x_1_next = std::get<2>(this->_jacobianOplus);
    auto& jacobian_wrt_x_2_next = std::get<3>(this->_jacobianOplus);

    jacobian_wrt_x_1_current = -weight_ * Eigen::Matrix<double,3,3>::Identity();
    jacobian_wrt_x_2_current = weight_ * Eigen::Matrix<double,3,3>::Identity();
    jacobian_wrt_x_1_next = weight_ * Eigen::Matrix<double,3,3>::Identity();
    jacobian_wrt_x_2_next = -weight_ * Eigen::Matrix<double,3,3>::Identity();
}
