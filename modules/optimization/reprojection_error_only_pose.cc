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

#include "reprojection_error_only_pose.h"

ReprojectionErrorOnlyPose::ReprojectionErrorOnlyPose(){}

bool ReprojectionErrorOnlyPose::read(std::istream& is){
    for (int i=0; i<2; i++){
        is >> _measurement[i];
    }
    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
        }
    return true;
}

bool ReprojectionErrorOnlyPose::write(std::ostream& os) const {

    for (int i=0; i<2; i++){
        os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++){
            os << " " <<  information()(i,j);
        }
    return os.good();
}

void ReprojectionErrorOnlyPose::computeError() {
    const g2o::VertexSE3Expmap* camera_pose_vertex =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector2d observation(_measurement);  //Observed point in the image

    Eigen::Vector3d landmark_camera = camera_pose_vertex->estimate().map(landmark_world_);
    _error = observation - calibration_->Project(landmark_camera);
}

void ReprojectionErrorOnlyPose::linearizeOplus() {
    g2o::VertexSE3Expmap * camera_pose_vertex =
            static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d landmark_camera = camera_pose_vertex->estimate().map(landmark_world_);

    double x = landmark_camera[0];
    double y = landmark_camera[1];
    double z = landmark_camera[2];

    Eigen::Matrix<double,2,3> projection_jacobian = -calibration_->ProjectionJacobian(landmark_camera);

    Eigen::Matrix<double,3,6> exponential_map_jacobian;
    exponential_map_jacobian << 0.f, z, -y, 1.f, 0.f, 0.f,
            -z , 0.f, x, 0.f, 1.f, 0.f,
            y ,  -x , 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXi = projection_jacobian * exponential_map_jacobian;
}
