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

#include "reprojection_error.h"

ReprojectionError::ReprojectionError() {};

bool ReprojectionError::read(std::istream& is){
    return true;
};

bool ReprojectionError::write(std::ostream& os) const{
    return true;
};

void ReprojectionError::computeError()  {
    const g2o::VertexSE3Expmap* pose_vertex = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const LandmarkVertex* landmark_vertex = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector2d observation(_measurement);      //Observed point in the image
    Eigen::Vector3d landmark_world_position = landmark_vertex->estimate();
    g2o::SE3Quat camera_transformation_world = pose_vertex->estimate();

    Eigen::Vector3d landmark_camera_position =
            camera_transformation_world.map(landmark_world_position);
    _error = observation - calibration_->Project(landmark_camera_position);
}

void ReprojectionError::linearizeOplus(){
    const g2o::VertexSE3Expmap* pose_vertex = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const LandmarkVertex* landmark_vertex = static_cast<const LandmarkVertex*>(_vertices[1]);

    g2o::SE3Quat camera_transformation_world = pose_vertex->estimate();
    Eigen::Vector3d landmark_world_position = landmark_vertex->estimate();
    Eigen::Vector3d landmark_camera_position =
            camera_transformation_world.map(landmark_world_position);

    Eigen::Matrix<double,2,3> projection_jacobian =
            -calibration_->ProjectionJacobian(landmark_camera_position);

    Eigen::Matrix<double,3,6> exponential_map_jacobian;
    exponential_map_jacobian << 0.f, landmark_camera_position.z(), -landmark_camera_position.y(), 1.f, 0.f, 0.f,
            -landmark_camera_position.z() , 0.f, landmark_camera_position.x(), 0.f, 1.f, 0.f,
            landmark_camera_position.y() ,  -landmark_camera_position.x() , 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXi = projection_jacobian * exponential_map_jacobian;
    _jacobianOplusXj = projection_jacobian * camera_transformation_world.rotation().toRotationMatrix();
}