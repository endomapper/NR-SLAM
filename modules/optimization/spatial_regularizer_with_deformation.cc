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

#include "spatial_regularizer_with_deformation.h"

#include "g2o/core/factory.h"

SpatialRegularizerWithDeformation::SpatialRegularizerWithDeformation(){};

bool SpatialRegularizerWithDeformation::read(std::istream& is){
    return true;
};

bool SpatialRegularizerWithDeformation::write(std::ostream& os) const{
    writeParamIds(os);
    os << " " << weight_ << " ";
    return writeInformationMatrix(os);
};

void SpatialRegularizerWithDeformation::computeError() {
    const LandmarkVertex* vertex_flow_1 = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* vertex_flow_2 = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector3d flow_1 = vertex_flow_1->estimate();
    Eigen::Vector3d flow_2 = vertex_flow_2->estimate();

    _error = weight_ * (flow_1 - flow_2);
}

void SpatialRegularizerWithDeformation::linearizeOplus() {
    _jacobianOplusXi = weight_ * Eigen::Matrix3d::Identity();
    _jacobianOplusXj = -weight_ * Eigen::Matrix3d::Identity();
}

G2O_REGISTER_TYPE(SPATIAL_REGULARIZER, SpatialRegularizerWithDeformation);
