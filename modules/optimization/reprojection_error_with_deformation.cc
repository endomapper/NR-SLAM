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

#include "reprojection_error_with_deformation.h"

#include "g2o/core/factory.h"

ReprojectionErrorWithDeformation::ReprojectionErrorWithDeformation() {};

bool ReprojectionErrorWithDeformation::read(std::istream& is){
    return true;
};

bool ReprojectionErrorWithDeformation::write(std::ostream& os) const{
    writeParamIds(os);
    g2o::internal::writeVector(os, measurement());
    os << " " << calibration_->ToString() << " ";
    return writeInformationMatrix(os);
};

void ReprojectionErrorWithDeformation::computeError()  {
    const g2o::VertexSE3Expmap* pose_vertex = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const LandmarkVertex* deformation_vertex = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector2d observation(_measurement);      //Observed point in the image
    g2o::SE3Quat Tcw = pose_vertex->estimate();
    Eigen::Vector3d deformation = deformation_vertex->estimate();

    Eigen::Vector3d landmark_camera = Tcw.map(deformation + landmark_world_);
    _error = observation - calibration_->Project(landmark_camera);
}

void ReprojectionErrorWithDeformation::linearizeOplus(){
    const g2o::VertexSE3Expmap* pose_vertex = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const LandmarkVertex* deformation_vertex = static_cast<const LandmarkVertex*>(_vertices[1]);

    g2o::SE3Quat Tcw = pose_vertex->estimate();
    Eigen::Vector3d deformation = deformation_vertex->estimate();

    Eigen::Vector3d landmark_deformed_world = deformation + landmark_world_;
    Eigen::Vector3d landmark_camera = Tcw.map(landmark_deformed_world);

    Eigen::Matrix<double,2,3> projection_jacobian = -calibration_->ProjectionJacobian(landmark_camera);

    Eigen::Matrix<double,3,6> exponential_map_jacobian;
    exponential_map_jacobian << 0.f, landmark_camera.z(), -landmark_camera.y(), 1.f, 0.f, 0.f,
            -landmark_camera.z() , 0.f, landmark_camera.x(), 0.f, 1.f, 0.f,
            landmark_camera.y() ,  -landmark_camera.x() , 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXi = projection_jacobian * exponential_map_jacobian;
    _jacobianOplusXj = projection_jacobian * Tcw.rotation().toRotationMatrix();
}

G2O_REGISTER_TYPE(REPROJECTION_ERROR, ReprojectionErrorWithDeformation);
