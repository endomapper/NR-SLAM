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

#ifndef NRSLAM_CAMERA_MODEL_H
#define NRSLAM_CAMERA_MODEL_H

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <eigen3/Eigen/Core>

class CameraModel {
public:
    CameraModel() {}


    // Constructs a camera model with the given calibration parameters.
    CameraModel(const std::vector<float>& calibration_parameters) :
        calibration_parameters_(calibration_parameters) {}


    // Projects a given 3D point into the image.
    virtual void Project(const Eigen::Vector3f& landmark_position,
                         Eigen::Vector2f& pixel_position) = 0;


    // Unprojects a given image 2D point into its projecting ray.
    virtual void Unproject(const Eigen::Vector2f& pixel_position,
                           Eigen::Vector3f& projecting_ray) = 0;


    // Computes the jacobian of the projection function.
    virtual void ProjectionJacobian(const Eigen::Vector3f& landmark_position,
                                    Eigen::Matrix<float,2,3>& projection_jacobian) = 0;

    // Computes the jacobian of the unprojection function.
    virtual void UnprojectionJacobian(const Eigen::Vector2f& pixel_position,
                                      Eigen::Matrix<float,3,2>& unprojection_jacobian) = 0;

    // Returns the projection function of the calibration model: a 3 by 3 matrix holding
    // the camera parameters.
    virtual Eigen::Matrix3f ToIntrinsicsMatrix() = 0;

    // Returns the calibration parameters.
    std::vector<float> GetParameters() {
        return calibration_parameters_;
    }

    // Gets the i-th calibration parameter of the camera model.
    float GetParameter(const int i) {
        return calibration_parameters_[i];
    }

    // Sets the i-th calibration parameter of the camera model.
    void SetParameter(const float p, const int i) {
        calibration_parameters_[i] = p;
    }

    // Returns the calibration model size.
    int getNumberOfParameters() {
        return calibration_parameters_.size();
    }

    // Method overloads to allow different data structures.
    cv::Point2f Project(const Eigen::Vector3f & landmark_position){
        Eigen::Vector2f pixel_position;

        this->Project(landmark_position.cast<float>(), pixel_position);

        cv::Point2f pixel_postion_opencv(pixel_position(0), pixel_position(1));
        return pixel_postion_opencv;
    }

    Eigen::Vector2d Project(Eigen::Vector3d & landmark_position){
        Eigen::Vector2f pixel_position;

        this->Project(landmark_position.cast<float>(), pixel_position);

        return pixel_position.cast<double>();
    }

    Eigen::Matrix<float,1,3> Unproject(const float pixel_u_coordinate, const float pixel_v_coordinate){
        Eigen::Vector2f pixel_position(pixel_u_coordinate, pixel_v_coordinate);
        Eigen::Vector3f projecting_ray;

        this->Unproject(pixel_position, projecting_ray);

        return projecting_ray;
    }

    Eigen::Matrix<float,1,3> Unproject(Eigen::Vector2f& pixel_position){
        Eigen::Vector3f projecting_ray;

        this->Unproject(pixel_position, projecting_ray);

        return projecting_ray;
    }

    Eigen::Matrix<float,1,3> Unproject(cv::Point2f pixel_position){
        Eigen::Vector2f pixel_position_eigen(pixel_position.x, pixel_position.y);
        Eigen::Vector3f projecting_ray;

        this->Unproject(pixel_position_eigen, projecting_ray);

        return projecting_ray;
    }

    Eigen::Matrix<float,2,3> ProjectionJacobian(Eigen::Vector3f &landmark_position){
        Eigen::Matrix<float,2,3> projection_jacobian;

        this->ProjectionJacobian(landmark_position, projection_jacobian);

        return projection_jacobian;
    }

    Eigen::Matrix<double,2,3> ProjectionJacobian(Eigen::Vector3d &landmark_position){
        Eigen::Matrix<float,2,3> projection_jacobian;

        this->ProjectionJacobian(landmark_position.cast<float>(), projection_jacobian);

        return projection_jacobian.cast<double>();
    }

    // Returns a string with the camera model parameters.
    std::string ToString() {
        std::string s;
        for (auto parameter : calibration_parameters_) {
            s += std::to_string(parameter) + " ";
        }

        return s;
    }


protected:
    // Vector storing the calibration parameters.
    std::vector<float> calibration_parameters_;
};


#endif //NRSLAM_CAMERA_MODEL_H
