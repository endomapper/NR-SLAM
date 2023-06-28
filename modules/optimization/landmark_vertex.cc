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

#include "landmark_vertex.h"

#include "g2o/stuff/misc.h"
#include "g2o/core/factory.h"

LandmarkVertex::LandmarkVertex() : BaseVertex<3, Eigen::Vector3d>() {}

bool LandmarkVertex::read(std::istream& is) {
    is >> _estimate(0 ), _estimate(1 ), _estimate(2 );
    return true;
}

bool LandmarkVertex::write(std::ostream& os) const {
    return g2o::internal::writeVector(os, estimate());
}

void LandmarkVertex::setToOriginImpl() {
    _estimate.fill(0);
}

void LandmarkVertex::oplusImpl(const double *update) {
    Eigen::Map<const Eigen::Vector3d> v(update);
    _estimate += v;
}

G2O_REGISTER_TYPE(VERTEX_LANDMARK, LandmarkVertex);
