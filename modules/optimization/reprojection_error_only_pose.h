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

#ifndef NRSLAM_REPROJECTION_ERROR_ONLY_POSE_H
#define NRSLAM_REPROJECTION_ERROR_ONLY_POSE_H

#include "calibration/camera_model.h"

#include "g2o/core/base_unary_edge.h"
#include "g2o/types/sba/vertex_se3_expmap.h"

class ReprojectionErrorOnlyPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d,g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorOnlyPose();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError();

    virtual void linearizeOplus();

    Eigen::Vector3d landmark_world_;
    std::shared_ptr<CameraModel> calibration_;
};


#endif //NRSLAM_REPROJECTION_ERROR_ONLY_POSE_H
