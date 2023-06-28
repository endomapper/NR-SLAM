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

#ifndef NRSLAM_POSITION_REGULARIZER_WITH_DEFORMATION_H
#define NRSLAM_POSITION_REGULARIZER_WITH_DEFORMATION_H

#include "optimization/landmark_vertex.h"

#include "g2o/core/base_binary_edge.h"

class PositionRegularizerWithDeformation : public g2o::BaseBinaryEdge<1, double,
        LandmarkVertex, LandmarkVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PositionRegularizerWithDeformation();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError();

    virtual void linearizeOplus();

    Eigen::Vector3d rest_position_1_, rest_position_2_;
    double k_;
};


#endif //NRSLAM_POSITION_REGULARIZER_WITH_DEFORMATION_H
