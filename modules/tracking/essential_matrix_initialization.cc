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

#include "essential_matrix_initialization.h"

#include "utilities/geometry_toolbox.h"

#include "absl/log/log.h"

using namespace std;

EssentialMatrixInitialization::EssentialMatrixInitialization(Options& options,
                                                             std::shared_ptr<CameraModel> calibration,
                                                             std::shared_ptr<ImageVisualizer> image_visualizer) :
        options_(options) {
    // Reserve memory.
    reference_keypoints_.reserve(options_.max_features);
    current_keypoints_.reserve(options_.max_features);

    reference_rays_.resize(options_.max_features, 3);
    current_rays_.resize(options_.max_features, 3);

    calibration_ = calibration;
    image_visualizer_ = image_visualizer;
}

void EssentialMatrixInitialization::ChangeReference(std::vector<cv::KeyPoint> &keypoints) {
    reference_keypoints_ = keypoints;
}

absl::Status EssentialMatrixInitialization::Initialize(const std::vector<cv::KeyPoint>& current_keypoints,
                                                       const std::vector<LandmarkStatus>& keypoint_statuses,
                                                       const int n_matches, Sophus::SE3f& camera_transform_world,
                                                       std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position) {
    if (n_matches < 8) {
        return absl::InternalError("Not enough matches");
    }
    // Set up input data.
    current_keypoints_ = current_keypoints;
    feature_tracks_statuses_ = keypoint_statuses;

    // Unproject tracked features.
    UnprojectTrackedFeatures();

    landmarks_position.resize(current_keypoints.size());
    fill(landmarks_position.begin(), landmarks_position.end(), absl::InternalError("Not triangulated"));

    // If enough features are tracked, try to find an Essential matrix with RANSAC.
    vector<bool> inliers_of_E(n_matches, false);
    int n_inliers_of_E;
    Eigen::Matrix3f E = FindEssentialWithRANSAC(n_matches, inliers_of_E, n_inliers_of_E);

    // Reconstruct the environment with the Essential matrix found.
    auto status =  ReconstructEnvironment(E, camera_transform_world, landmarks_position, n_inliers_of_E, inliers_of_E);

    if (image_visualizer_)
        image_visualizer_->DrawFeatures(current_keypoints_, landmarks_position);

    return status;
}

int EssentialMatrixInitialization::ComputeMaxTries(const float inlier_fraction, const float success_likelihood){
    return log(1 - success_likelihood) /
           log(1 - pow(inlier_fraction, options_.min_sample_set_size));
}

void EssentialMatrixInitialization::UnprojectTrackedFeatures() {
    ransac_idx_to_keypoint_idx_.clear();
    reference_keypoints_tracked_.clear();

    int n_tracked_points = 0;

    for(int idx = 0; idx < feature_tracks_statuses_.size(); idx++){
        if(feature_tracks_statuses_[idx] == TRACKED){
            reference_rays_.block(n_tracked_points, 0, 1, 3) =
                    calibration_->Unproject(reference_keypoints_[idx].pt.x, reference_keypoints_[idx].pt.y).normalized();
            current_rays_.block(n_tracked_points, 0, 1, 3) =
                    calibration_->Unproject(current_keypoints_[idx].pt.x, current_keypoints_[idx].pt.y).normalized();

            reference_keypoints_tracked_.push_back(reference_keypoints_[idx].pt);

            ransac_idx_to_keypoint_idx_.push_back(idx);

            n_tracked_points++;
        }
    }
}

Eigen::Matrix3f EssentialMatrixInitialization::FindEssentialWithRANSAC(const int n_matches,
                                                                       vector<bool>& inliers, int& n_inliers) {
    int best_score = 0;
    vector<bool> inliers_best_model;
    inliers_best_model.reserve(n_matches);
    Eigen::Matrix<float,3,3> best_E;

    srand(4);

    // Cluster data for random selection.
    const int n_clusters = options_.min_sample_set_size;
    const int n_attemps = 3;
    auto termination_criteria = cv::TermCriteria(cv::TermCriteria::EPS, 10, 1.0);
    vector<int> labels;
    vector<cv::Point2f> centers;
    cv::kmeans(reference_keypoints_tracked_, n_clusters, labels, termination_criteria,
               n_attemps, cv::KMEANS_PP_CENTERS, centers);

    // Assign indices to clusters with the given labels.
    vector<vector<int>> clusters = vector<vector<int>>(n_clusters);
    for(int idx = 0; idx < labels.size(); idx++){
        clusters[labels[idx]].push_back(idx);
    }

    // Compute number of RANSAC iterations
    const float inlier_fraction = 0.8;
    const float sucess_likelihood = 0.95;
    int max_iterations = ComputeMaxTries(inlier_fraction, sucess_likelihood);
    int current_iteration = 0;

    // Do all iterations.
    while(current_iteration < max_iterations){
        current_iteration++;

        // Get minimum sample set of data.
        Eigen::Matrix<float,8,3> reference_rays_sample, current_rays_sample;
        for(int idx = 0; idx < n_clusters; idx++){
            random_shuffle(clusters[idx].begin(), clusters[idx].end());
            int feature_idx = clusters[idx][0];

            reference_rays_sample.block<1, 3>(idx, 0) = reference_rays_.block(feature_idx, 0, 1, 3);
            current_rays_sample.block<1, 3>(idx, 0) = current_rays_.block(feature_idx, 0, 1, 3);
        }

        // Compute model with the sample.
        Eigen::Matrix<float,3,3> E = ComputeE(reference_rays_sample, current_rays_sample);

        // Get score and inliers for the computed model.
        int score = ComputeScoreAndInliers(n_matches, E, inliers);
        if(score > best_score){
            best_score = score;
            inliers_best_model = inliers;
            best_E = E;
        }
    }

    // Recompute E with all the inliers.
    Eigen::MatrixXf best_model_reference_rays(best_score, 3);
    Eigen::MatrixXf best_model_current_rays(best_score, 3);

    int current_idx = 0;
    for(int idx = 0; idx < inliers_best_model.size(); idx++){
        if(inliers_best_model[idx]){
            best_model_reference_rays.block<1,3>(current_idx,0) = reference_rays_.block(idx, 0, 1, 3);
            best_model_current_rays.block<1,3>(current_idx,0) = current_rays_.block(idx, 0, 1, 3);
            current_idx++;
        }
    }

    int score = ComputeScoreAndInliers(n_matches, best_E, inliers);
    n_inliers = score;

    return best_E;
}

Eigen::Matrix3f EssentialMatrixInitialization::ComputeE(Eigen::Matrix<float, 8, 3> &reference_rays_sample,
                                                        Eigen::Matrix<float, 8, 3> &current_rays_sample) {
    // Compute a first estimation of the Essential matrix.
    Eigen::Matrix<float, 8, 9> A;
    for(int idx = 0; idx < 8; idx++){
        A.block<1,3>(idx, 0) = reference_rays_sample.row(idx) * current_rays_sample(idx, 0);
        A.block<1,3>(idx, 3) = reference_rays_sample.row(idx) * current_rays_sample(idx, 1);
        A.block<1,3>(idx, 6) = reference_rays_sample.row(idx) * current_rays_sample(idx, 2);
    }

    Eigen::JacobiSVD<Eigen::Matrix<float, 8, 9>> svd_solver(A, Eigen::ComputeFullV);
    svd_solver.computeV();
    Eigen::Matrix<float, 9, 1> eigen_vectors = svd_solver.matrixV().col(8);
    Eigen::Matrix3f E;
    E.row(0) = eigen_vectors.block<3, 1>(0, 0).transpose();
    E.row(1) = eigen_vectors.block<3, 1>(3, 0).transpose();
    E.row(2) = eigen_vectors.block<3, 1>(6, 0).transpose();

    // Force eigen values into the Essential matrix
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_essential_matrix(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f eigen_values;
    eigen_values << 1, 1, 0;
    Eigen::Matrix<float,3,3> Ef = svd_essential_matrix.matrixU() * eigen_values.asDiagonal() *
            svd_essential_matrix.matrixV().transpose();

    return -Ef;
}

Eigen::Matrix3f EssentialMatrixInitialization::RefineSolution(Eigen::MatrixXf &reference_rays,
                                                              Eigen::MatrixXf &current_rays) {
    int data_size = reference_rays.rows();
    Eigen::MatrixXf A(data_size, 9);
    for(int idx = 0; idx < data_size; idx++){
        A.block<1, 3>(idx, 0) = reference_rays.row(idx) * current_rays(idx,0);
        A.block<1, 3>(idx, 3) = reference_rays.row(idx) * current_rays(idx,1);
        A.block<1, 3>(idx, 6) = reference_rays.row(idx) * current_rays(idx,2);
    }

    Eigen::BDCSVD<Eigen::MatrixXf> svd_solver(A,Eigen::ComputeFullV);
    svd_solver.computeV();
    Eigen::MatrixXf eigen_vectors = svd_solver.matrixV().col(8);
    Eigen::Matrix3f E;
    E.row(0) = eigen_vectors.block<3, 1>(0, 0).transpose();
    E.row(1) = eigen_vectors.block<3, 1>(3, 0).transpose();
    E.row(2) = eigen_vectors.block<3, 1>(6, 0).transpose();

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_essential_matrix(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f eigen_values;
    eigen_values << 1, 1, 0;
    Eigen::Matrix<float, 3, 3> Ef = svd_essential_matrix.matrixU() * eigen_values.asDiagonal() *
            svd_essential_matrix.matrixV().transpose();

    return -Ef;
}


int EssentialMatrixInitialization::ComputeScoreAndInliers(const int n_matched,
                                                          Eigen::Matrix<float,3,3>& E,
                                                          std::vector<bool>& inliers) {
    Eigen::MatrixXf reference_rays_transformed = (E * reference_rays_.block(0, 0, n_matched, 3)
            .transpose()).transpose().rowwise().normalized();

    auto errors = (M_PI / 2 - (reference_rays_transformed.array() *
            current_rays_.block(0, 0, n_matched, 3).rowwise().normalized().array())
            .rowwise().sum().acos()).abs() < options_.epipolar_threshold;

    int score = 0;
    fill(inliers.begin(), inliers.end(), false);
    for (int idx = 0; idx < n_matched; idx++){
        if (errors(idx)) {
            inliers[idx] = true;
            score++;
        }
    }

    return score;
}

absl::Status EssentialMatrixInitialization::ReconstructEnvironment(
        Eigen::Matrix3f& E, Sophus::SE3f& camera_transform_world,
        std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
        int& n_inliers, std::vector<bool>& essential_matrix_inliers) {
    // Compute rays of the inliers points.
    Eigen::MatrixXf reference_rays(n_inliers, 3);
    Eigen::MatrixXf current_rays(n_inliers, 3);
    int current_idx = 0;
    for(int idx = 0; idx < essential_matrix_inliers.size(); idx++){
        if(essential_matrix_inliers[idx]){
            reference_rays.row(current_idx) =
                    calibration_->Unproject(reference_keypoints_[idx].pt.x, reference_keypoints_[idx].pt.y).normalized();
            current_rays.row(current_idx) =
                    calibration_->Unproject(current_keypoints_[idx].pt.x, current_keypoints_[idx].pt.y).normalized();

            current_idx++;
        }
    }

    //Reconstruct camera poses
    ReconstructCameras(E, camera_transform_world, reference_rays, current_rays);

    //Reconstruct 3D points (try with the 2 possible translations)
    return ReconstructPoints(camera_transform_world, landmarks_position, essential_matrix_inliers);
}

void EssentialMatrixInitialization::ReconstructCameras(Eigen::Matrix3f &E, Sophus::SE3f &camera_transform_world,
                                                       Eigen::MatrixXf& rays_1, Eigen::MatrixXf& rays_2) {
    // Decompose E into 2 rotation hypotheses (R_1 and R_2) and a translation vector.
    Eigen::Matrix3f R_1, R_2, R_good;
    Eigen::Vector3f t;
    DecomposeEssentialMatrix(E, R_1, R_2, t);

    // Choose the smallest rotation.
    R_good = (R_2.trace() > R_1.trace()) ? R_2 : R_1;

    // Get correct translation.
    float away = ((R_good * rays_1.transpose() - rays_2.transpose()).array() *
                  (rays_2.transpose().colwise() - t).array()).colwise().sum().sign().sum();

    t = (signbit(away)) ? -t : t;

    camera_transform_world = Sophus::SE3f(R_good, t);
}

void EssentialMatrixInitialization::DecomposeEssentialMatrix(Eigen::Matrix3f &E, Eigen::Matrix3f &R_1,
                                                             Eigen::Matrix3f &R_2, Eigen::Vector3f &t) {
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    R_1 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();
    if(R_1.determinant() < 0)
        R_1 = -R_1;

    R_2 = svd.matrixU() * W * svd.matrixV().transpose();
    if(R_2.determinant() < 0)
        R_2 = -R_2;

    t = svd.matrixU().col(2).normalized();
}

absl::Status EssentialMatrixInitialization::ReconstructPoints(const Sophus::SE3f &camera_transform_world,
                                                      std::vector<absl::StatusOr<Eigen::Vector3f>>&landmarks_position,
                                                      std::vector<bool>& essential_matrix_inliers) {
    vector<float> landmarks_parallax;
    int n_triangulated = 0;
    int n_parallax = 0, n_reprojection_error_1 = 0, n_reprojection_error_2 = 0, n_triangulation_error = 0;
    int n_depth_1 = 0, n_depth_2 = 0;
    int N = 0;

    Eigen::Vector3f world_t_camera = camera_transform_world.inverse().translation();

    for (int idx = 0; idx < essential_matrix_inliers.size(); idx++){
        if( essential_matrix_inliers[idx]){
            N++;
            // Unproject KeyPoints to rays.
            Eigen::Vector3f reference_ray =
                    calibration_->Unproject(reference_keypoints_[idx].pt.x, reference_keypoints_[idx].pt.y).normalized();
            Eigen::Vector3f current_ray =
                    calibration_->Unproject(current_keypoints_[idx].pt.x, current_keypoints_[idx].pt.y).normalized();

            // Triangulate point.
            auto landmark_position_status =
                    TriangulateMidPoint(reference_ray, current_ray, Sophus::SE3f(), camera_transform_world);
            if (!landmark_position_status.ok()) {
                landmarks_position[idx] = absl::InternalError("Internal triangulation error.");
                n_triangulation_error++;
                continue;
            }

            Eigen::Vector3f landmark_position = *landmark_position_status;

            // Check the parallax of the triangulated point.
            Eigen::Vector3f normal_1 = landmark_position;
            Eigen::Vector3f normal_2 = landmark_position - world_t_camera;
            float parallax = RaysParallax(normal_1, normal_2);

            if(parallax < options_.radians_per_pixel * 5.f){
                landmarks_position[idx] = absl::InternalError("Low parallax error.");
                n_parallax++;
                continue;
            }

            // Check that the point has been triangulated in front of the first camera (positive depth).
            if(landmark_position(2) < 0.0f){
                landmarks_position[idx] = absl::InternalError("Negative depth at first camera.");
                n_depth_1++;
                continue;
            }

            // Check Reprojection error.
            cv::Point2f projected_landmark_1 = calibration_->Project(landmark_position);

            if(SquaredReprojectionError(reference_keypoints_[idx].pt, projected_landmark_1) > 5.991){
                landmarks_position[idx] = absl::InternalError("High reprojection error at first camera.");
                n_reprojection_error_1++;
                continue;
            }

            Eigen::Vector3f landmark_position_camera2 = camera_transform_world * landmark_position;
            if(landmark_position_camera2(2) < 0.0f){
                landmarks_position[idx] = absl::InternalError("Negative depth at second camera.");
                n_depth_2++;
                continue;
            }

            cv::Point2f projected_landmark_2 = calibration_->Project(landmark_position_camera2);

            if(SquaredReprojectionError(current_keypoints_[idx].pt, projected_landmark_2) > 5.991){
                landmarks_position[idx] = absl::InternalError("High reprojection error at second camera.");
                n_reprojection_error_2++;
                continue;
            }

            landmarks_position[idx] = landmark_position;

            n_triangulated++;
            landmarks_parallax.push_back(parallax);

        }
    }

    if(n_triangulated < 100){
        return absl::InternalError("Not enough triangulated landmarks");
    }

    if(n_parallax > N * 0.25){
        return absl::InternalError("Not enough triangulated landmarks");
    }

    return absl::OkStatus();
}
