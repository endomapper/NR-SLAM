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
#include "kannala_brandt_8.h"

#include "absl/log/log.h"

using namespace std;

#define fx calibration_parameters_[0]
#define fy calibration_parameters_[1]
#define cx calibration_parameters_[2]
#define cy calibration_parameters_[3]
#define k0 calibration_parameters_[4]
#define k1 calibration_parameters_[5]
#define k2 calibration_parameters_[6]
#define k3 calibration_parameters_[7]

void KannalaBrandt8::Project(const Eigen::Vector3f& landmark_position,
                             Eigen::Vector2f& pixel_position) {
    const float x2_plus_y2 = landmark_position[0] * landmark_position[0]
            + landmark_position[1] * landmark_position[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), landmark_position[2]);
    const float psi = atan2f(landmark_position[1], landmark_position[0]);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + k0 * theta3 + k1 * theta5
                    + k2 * theta7 + k3 * theta9;

    pixel_position(0) = calibration_parameters_[0] * r * cos(psi) + calibration_parameters_[2];
    pixel_position(1) = calibration_parameters_[1] * r * sin(psi) + calibration_parameters_[3];
}

void KannalaBrandt8::Unproject(const Eigen::Vector2f& pixel_position,
                               Eigen::Vector3f& projecting_ray) {
    // Use Newthon method to solve for theta with good precision (err ~ e-6).
    cv::Point2f pw((pixel_position[0] - calibration_parameters_[2]) / calibration_parameters_[0],
                   (pixel_position[1] - calibration_parameters_[3]) / calibration_parameters_[1]);
    float scale = 1.f;
    const float theta_d = sqrtf(pw.x * pw.x + pw.y * pw.y);

    float th;
    if (theta_d > 1e-8) {
        // Compensate distortion iteratively.
        float theta = theta_d;

        for (int j = 0; j < 10; j++) {
            float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 =
                    theta4 * theta4;
            float k0_theta2 = k0 * theta2, k1_theta4 = k1 * theta4;
            float k2_theta6 = k2 * theta6, k3_theta8 = k3 * theta8;
            float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                              (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
            theta = theta - theta_fix;
            if (fabsf(theta_fix) < precision_)
                break;
        }
        scale = std::tan(theta) / theta_d;

        th = theta;
    }

    projecting_ray[0] = sin(th) * pw.x/ theta_d;
    projecting_ray[1] = sin(th) * pw.y/ theta_d;
    projecting_ray[2] = cos(th);
}

void KannalaBrandt8::ProjectionJacobian(const Eigen::Vector3f& landmark_position,
                                        Eigen::Matrix<float,2,3>& projection_jacobian) {
    float x2 = landmark_position[0] * landmark_position[0];
    float y2 = landmark_position[1] * landmark_position[1];
    float z2 = landmark_position[2] * landmark_position[2];
    float r2 = x2 + y2;
    float r = sqrt(r2);
    float r3 = r2 * r;
    float theta = atan2(r, landmark_position[2]);

    float theta2 = theta * theta, theta3 = theta2 * theta;
    float theta4 = theta2 * theta2, theta5 = theta4 * theta;
    float theta6 = theta2 * theta4, theta7 = theta6 * theta;
    float theta8 = theta4 * theta4, theta9 = theta8 * theta;

    float f = theta + theta3 * k0 + theta5 * k1 + theta7 * k2 +
              theta9 * k3;
    float fd = 1 + 3 * k0 * theta2 + 5 * k1 * theta4 + 7 * k2 * theta6 +
               9 * k3 * theta8;

    projection_jacobian(0,0) = fx * (fd * landmark_position[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
    projection_jacobian(0,1) = fx * (fd * landmark_position[2] * landmark_position[1] * landmark_position[0] /
            (r2 * (r2 + z2)) - f * landmark_position[1] * landmark_position[0] / r3);
    projection_jacobian(0,2) = -fx * fd * landmark_position[0] / (r2 + z2);

    projection_jacobian(1,0) = fy * (fd * landmark_position[2] * landmark_position[1] * landmark_position[0] /
            (r2 * (r2 + z2)) - f * landmark_position[1] * landmark_position[0] / r3);
    projection_jacobian(1,1) = fy * (fd * landmark_position[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);
    projection_jacobian(1,2) = -fy * fd * landmark_position[1] / (r2 + z2);
}

void KannalaBrandt8::UnprojectionJacobian(const Eigen::Vector2f& pixel_position,
                                          Eigen::Matrix<float,3,2>& unprojection_jacobian) {
    LOG(ERROR) << "UnprojectionJacobian not implmented yet.";
}

Eigen::Matrix3f KannalaBrandt8::ToIntrinsicsMatrix() {
    Eigen::Matrix3f intrinsics_matrix = Eigen::Matrix3f::Identity();
    intrinsics_matrix(0,0) = fx;
    intrinsics_matrix(0,2) = cx;
    intrinsics_matrix(1,1) = fy;
    intrinsics_matrix(1,2) = cy;

    return intrinsics_matrix;
}