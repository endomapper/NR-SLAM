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

#include "spatial_regularizer_fixed.h"

SpatialRegularizerFixed::SpatialRegularizerFixed(){};

bool SpatialRegularizerFixed::read(std::istream& is){
    return true;
}

bool SpatialRegularizerFixed::write(std::ostream& os) const {
    return true;
}

void SpatialRegularizerFixed::computeError() {
    const LandmarkVertex* vertex_flow = static_cast<const LandmarkVertex*>(_vertices[0]);

    Eigen::Vector3d flow_1 = vertex_flow->estimate();
    Eigen::Vector3d flow_2 = flow_fixed->estimate();

    _error = weight_ * (flow_1 - flow_2);
}

void SpatialRegularizerFixed::linearizeOplus() {
    _jacobianOplusXi = weight_ * Eigen::Matrix3d::Identity();
}
