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

#include "g2o_optimization.h"

#include "optimization/landmark_vertex.h"
#include "optimization/position_regularizer.h"
#include "optimization/position_regularizer_with_deformation.h"
#include "optimization/reprojection_error.h"
#include "optimization/reprojection_error_only_deformation.h"
#include "optimization/reprojection_error_only_pose.h"
#include "optimization/reprojection_error_with_deformation.h"
#include "optimization/spatial_regularizer.h"
#include "optimization/spatial_regularizer_with_deformation.h"
#include "optimization/spatial_regularizer_with_observation.h"
#include "optimization/spatial_regularizer_fixed.h"
#include "utilities/geometry_toolbox.h"

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/container/btree_set.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"

using namespace std;

void CameraPoseOptimization(Frame& frame, const Sophus::SE3f& previous_camera_transform_world) {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
            g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    const float th_huber_2dof_squared = 5.99;
    const float th_huber_2dof = sqrt(th_huber_2dof_squared);

    // Set camera pose vertex.
    g2o::VertexSE3Expmap* camera_pose_vertex = new g2o::VertexSE3Expmap();
    Sophus::SE3f camera_transform_world = frame.CameraTransformationWorld();
    camera_pose_vertex->setEstimate(g2o::SE3Quat(
            camera_transform_world.unit_quaternion().cast<double>(),
            camera_transform_world.translation().cast<double>()));
    camera_pose_vertex->setId(0);

    optimizer.addVertex(camera_pose_vertex);

    vector<cv::KeyPoint> keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    vector<Eigen::Vector3f> landmark_positions = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    vector<ID> landmark_ids = frame.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    vector<ReprojectionErrorOnlyPose*> reprojection_error_edges(keypoints.size(), nullptr);

    for(int idx = 0; idx < keypoints.size(); idx++){
        cv::Point2f pixel_coordinates = keypoints[idx].pt;
        Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);

        ReprojectionErrorOnlyPose* edge = new ReprojectionErrorOnlyPose();

        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(observation);
        edge->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
        edge->setRobustKernel(robust_kernel);
        robust_kernel->setDelta(th_huber_2dof);

        edge->calibration_ = frame.GetCalibration();
        edge->landmark_world_ = landmark_positions[idx].cast<double>();

        optimizer.addEdge(edge);
        reprojection_error_edges[idx] = edge;
    }

    vector<int> iterations = {10, 10, 10};
    vector<bool> inliers(keypoints.size(), true);

    for(int iteration = 0; iteration < iterations.size(); iteration++){
        // Reset position to the initial seed.
        camera_pose_vertex->setEstimate(g2o::SE3Quat(
                camera_transform_world.unit_quaternion().cast<double>(),
                camera_transform_world.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(iterations[iteration]);

        for(int idx = 0; idx < reprojection_error_edges.size(); idx++){
            ReprojectionErrorOnlyPose* edge = reprojection_error_edges[idx];
            if(!edge)
                continue;

            if(!inliers[idx]) {
                edge->computeError();
            }

            const float chi_squared = edge->chi2();

            if(chi_squared > th_huber_2dof_squared) {
                inliers[idx] = false;
                edge->setLevel(1);
            }
            else {
                inliers[idx] = true;
                edge->setLevel(0);
            }

            // Deactivate robust kernel after 2 iterations as we should have removed
            // all the inlier observations.
            if(iteration == 2){
                edge->setRobustKernel(0);
            }
        }
    }

    // Recover the optimized camera pose.
    frame.MutableCameraTransformationWorld() = Sophus::SE3f(
            camera_pose_vertex->estimate().to_homogeneous_matrix().cast<float>());
}

absl::flat_hash_set<ID> CameraPoseAndDeformationOptimization(Frame& current_frame,
                                                     std::shared_ptr<Map> map,
                                                     const Sophus::SE3f& previous_camera_transform_world,
                                                     const float scale) {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set camera vertex
    g2o::VertexSE3Expmap* camera_pose_vertex = new g2o::VertexSE3Expmap();
    Sophus::SE3f camera_transform_world = current_frame.CameraTransformationWorld();
    camera_pose_vertex->setEstimate(g2o::SE3Quat(
            camera_transform_world.unit_quaternion().cast<double>(),
            camera_transform_world.translation().cast<double>()));
    camera_pose_vertex->setId(0);
    camera_pose_vertex->setFixed(false);

    optimizer.addVertex(camera_pose_vertex);

    vector<cv::KeyPoint> keypoints = current_frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    vector<Eigen::Vector3f> landmark_positions = current_frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    vector<ID> mappoints_ids = current_frame.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
    absl::flat_hash_map<ID, int> mappoint_id_to_index;
    const int points_in_optimization = keypoints.size();

    auto regularization_graph = map->GetRegularizationGraph();

    // Set point vertices.
    vector<LandmarkVertex*> deformation_vertices(points_in_optimization, nullptr);
    for (int idx = 0; idx < points_in_optimization; idx++) {
        deformation_vertices[idx] = new LandmarkVertex();
        deformation_vertices[idx]->setId(idx + 1);
        deformation_vertices[idx]->setToOrigin();

        optimizer.addVertex(deformation_vertices[idx]);

        mappoint_id_to_index[mappoints_ids[idx]] = idx;
    }

    // Set error terms.
    const int regularizers_per_point = 10;

    const float th_huber_2dof_squared = 5.99;
    const float th_huber_2dof = sqrt(th_huber_2dof_squared);

    const float th_huber_3dof_squared = 0.584;
    const float th_huber_3dof = sqrt(th_huber_3dof_squared);

    const float sigma_reprojection = 0.5;   // pixels.
    const float info_reprojection = 1.0f / (sigma_reprojection * sigma_reprojection);

    float sigma_position = 0.1f;
    float info_position = 1.0f / (sigma_position * sigma_position);

    const float sigma_spatial = 0.1 * scale;   // mm.
    const float info_spatial = 1.0f / (sigma_spatial * sigma_spatial);

    vector<absl::flat_hash_map<int, SpatialRegularizerWithDeformation*>> spatial_regularizers(points_in_optimization);
    vector<absl::flat_hash_map<int, PositionRegularizerWithDeformation*>> position_regularizers(points_in_optimization);
    vector<ReprojectionErrorWithDeformation*> reprojection_errors(points_in_optimization);

    int n_reprojection_edges = 0;
    int n_spatial_edges = 0;
    int n_position_edges = 0;

    vector<vector<ID>> connected_mappoint_ids(points_in_optimization);

    absl::btree_set<ID> lost_mappoint_ids_ordered;
    absl::flat_hash_set<ID> lost_mappoint_ids;
    for (int idx = 0; idx < points_in_optimization; idx++) {
        // Set reprojection error.
        ReprojectionErrorWithDeformation* reprojection_error = new ReprojectionErrorWithDeformation();

        cv::Point2f pixel_coordinates = keypoints[idx].pt;
        Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);

        reprojection_error->setMeasurement(observation);

        reprojection_error->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        reprojection_error->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(idx+1)));

        reprojection_error->setInformation(Eigen::Matrix2d::Identity() * info_reprojection);

        g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
        robust_kernel->setDelta(th_huber_2dof);
        reprojection_error->setRobustKernel(robust_kernel);

        reprojection_error->calibration_ = current_frame.GetCalibration();
        reprojection_error->landmark_world_ = landmark_positions[idx].cast<double>();

        optimizer.addEdge(reprojection_error);
        reprojection_errors[idx] = reprojection_error;

        n_reprojection_edges++;

        // Set spatial regularizers.
        const ID mappoint_id = mappoints_ids[idx];
        auto regularization_edges =
                regularization_graph->GetEdges(mappoint_id);

        int n_regularizers = 0;
        for(const auto& [mappoint_id_other, regularization_edge] : regularization_edges) {
            // Check if there is already enough regularizers or if the connection is good.
            if (n_regularizers > regularizers_per_point ||
                regularization_edge->status == RegularizationGraph::BAD) {
                break;
            }

            // Check that the connected point is also being optimized.
            if (!current_frame.MapPointIdToIndex().contains(mappoint_id_other) ||
                current_frame.LandmarkStatuses()[current_frame.MapPointIdToIndex().at(mappoint_id_other)] != TRACKED_WITH_3D) {
                if (current_frame.MapPointIdToIndex().contains(mappoint_id_other) &&
                    current_frame.LandmarkStatuses()[current_frame.MapPointIdToIndex().at(mappoint_id_other)] != JUST_TRIANGULATED) {
                    lost_mappoint_ids.insert(mappoint_id_other);
                    lost_mappoint_ids_ordered.insert(mappoint_id_other);
                }

                continue;
            }

            // Check if this regularizer has already been inserted in the optimization.
            const int idx_other = mappoint_id_to_index[mappoint_id_other];
            if (spatial_regularizers[idx].contains(idx_other)) {
                continue;
            }

            // Set spatial regularizer.
            auto spatial_regularizer = new SpatialRegularizerWithDeformation();
            spatial_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx + 1)));
            spatial_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx_other + 1)));

            Eigen::Matrix3d spatial_information_matrix = Eigen::Matrix3d::Identity() * info_spatial;
            spatial_regularizer->setInformation(spatial_information_matrix);

            g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(th_huber_3dof);
            spatial_regularizer->setRobustKernel(robust_kernel);

            spatial_regularizer->weight_ = regularization_edge->weight;

            optimizer.addEdge(spatial_regularizer);

            spatial_regularizers[idx][idx_other] = spatial_regularizer;
            spatial_regularizers[idx_other][idx] = spatial_regularizer;

            connected_mappoint_ids[idx].push_back(mappoint_id_other);
            connected_mappoint_ids[idx_other].push_back(mappoint_id);
            n_spatial_edges++;

            n_regularizers++;

            // Set position regularizer.
            PositionRegularizerWithDeformation* position_regularizer = new PositionRegularizerWithDeformation();
            position_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx + 1)));
            position_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx_other + 1)));

            position_regularizer->setMeasurement(regularization_edge->first_distance);

            Eigen::Matrix<double, 1, 1> position_information_matrix =
                    Eigen::Matrix<double, 1, 1>::Identity() * info_position;
            position_regularizer->setInformation(position_information_matrix);

            position_regularizer->rest_position_1_ = landmark_positions[idx].cast<double>();
            position_regularizer->rest_position_2_ = landmark_positions[idx_other].cast<double>();

            g2o::RobustKernelHuber* robust_kernel_position = new g2o::RobustKernelHuber;
            robust_kernel_position->setDelta(th_huber_3dof);
            position_regularizer->setRobustKernel(robust_kernel_position);

            position_regularizer->k_ = 1.1f;

            optimizer.addEdge(position_regularizer);

            n_position_edges++;

            position_regularizers[idx][idx_other] = position_regularizer;
            position_regularizers[idx_other][idx] = position_regularizer;
        }
    }
    vector<int> iterations = {10, 10};
    vector<bool> inliers(points_in_optimization, true);

    for (int iteration = 0; iteration < iterations.size(); iteration++){
        int n_good_regularizers = 0;

        // Reset estimations.
        camera_pose_vertex->setEstimate(g2o::SE3Quat(
                camera_transform_world.unit_quaternion().cast<double>(),
                camera_transform_world.translation().cast<double>()));

        for(auto vertex : deformation_vertices){
            if(!vertex) {
                continue;
            } else{
                vertex->setToOrigin();
            }
        }

        optimizer.initializeOptimization(0);
        optimizer.optimize(iterations[iteration]);

        // Check reprojection errors.
        for(int idx = 0; idx < points_in_optimization; idx++){
            ReprojectionErrorWithDeformation* reprojection_error = reprojection_errors[idx];

            reprojection_error->computeError();

            const float chi_squared = reprojection_error->chi2();

            if(chi_squared > th_huber_2dof_squared) {
                inliers[idx] = false;

                reprojection_error->setLevel(1);
                for (auto& [idx_other, spatial_regularizer] : spatial_regularizers[idx]) {
                    spatial_regularizer->setLevel(1);
                }
            } else {
                inliers[idx] = true;

                reprojection_error->setLevel(0);
                for (auto& [idx_other, spatial_regularizer] : spatial_regularizers[idx]) {
                    spatial_regularizer->setLevel(0);
                }
            }

            // Check spatial regularizers.
            for (auto& [idx_other, spatial_regularizer] : spatial_regularizers[idx]) {
                spatial_regularizer->computeError();

                if(spatial_regularizer->chi2() > th_huber_3dof_squared) {
                    spatial_regularizer->setLevel(1);
                } else {
                    spatial_regularizer->setLevel(0);
                }
            }
        }
    }

    // Recover the optimized camera pose.
    current_frame.MutableCameraTransformationWorld() = Sophus::SE3f(
            camera_pose_vertex->estimate().to_homogeneous_matrix().cast<float>());

    vector<float> deformation_magnitudes;
    for(int idx = 0; idx < points_in_optimization; idx++) {
        LandmarkVertex* deformation_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx+1));
        Eigen::Vector3f deformation = deformation_vertex->estimate().cast<float>();
        deformation_magnitudes.push_back(deformation.norm());
    }

    vector<float> sorted_deformation_magnitudes = deformation_magnitudes;
    sort(sorted_deformation_magnitudes.begin(), sorted_deformation_magnitudes.end());
    float interquartileRange = sorted_deformation_magnitudes[(int)(sorted_deformation_magnitudes.size() * 0.75f)] -
            sorted_deformation_magnitudes[(int)(sorted_deformation_magnitudes.size() * 0.25f)];
    float q1 = sorted_deformation_magnitudes[(int)(sorted_deformation_magnitudes.size() * 0.25f)];
    float q3 = sorted_deformation_magnitudes[(int)(sorted_deformation_magnitudes.size() * 0.75f)];

    float th_ = 1.5f * interquartileRange;

    // Update point positions.
    for(int idx = 0; idx < points_in_optimization; idx++) {
        ReprojectionErrorWithDeformation* reprojection_error = reprojection_errors[idx];
        reprojection_error->computeError();

        int index_in_frame = current_frame.MapPointIdToIndex().at(mappoints_ids[idx]);

        const float chi_squared = reprojection_error->chi2();
        if(chi_squared > th_huber_2dof_squared) {
            inliers[idx] = false;

            current_frame.LandmarkStatuses()[index_in_frame] = TRACKED;
        }

        LandmarkVertex* deformation_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx+1));
        Eigen::Vector3f deformation = deformation_vertex->estimate().cast<float>();

        if (deformation.norm() >= q3 + th_) {
            current_frame.LandmarkStatuses()[index_in_frame] = TRACKED;
            continue;
        }

        deformation_vertex->setFixed(true);

        Eigen::Vector3f previous_landmark_position = landmark_positions[idx];
        Eigen::Vector3f current_landmark_position = deformation + previous_landmark_position;

        current_frame.LandmarkPositions()[index_in_frame] = current_landmark_position;

        map->GetMapPoint(mappoints_ids[idx])->SetLastWorldPosition(current_landmark_position);

        ID mappoint_id = mappoints_ids[idx];
    }

    const int median_idx = deformation_magnitudes.size() / 2;
    nth_element(deformation_magnitudes.begin(), deformation_magnitudes.begin() + median_idx,
                deformation_magnitudes.end());

    current_frame.SetDeformationMaginitud(deformation_magnitudes[median_idx]);

    // Update regularization graph.
    for(int idx = 0; idx < points_in_optimization; idx++) {
        if (!inliers[idx]) {
            continue;
        }

        ID mappoint_id = mappoints_ids[idx];

        int index_in_frame = current_frame.MapPointIdToIndex().at(mappoints_ids[idx]);
        Eigen::Vector3f landmark_position = current_frame.LandmarkPositions()[index_in_frame];

        int good_connections = regularization_graph->UpdateVertex(mappoint_id);

        if (good_connections < regularizers_per_point * 0.5) {
            LOG(INFO) << "Removing features because graph error: ";
            current_frame.LandmarkStatuses()[index_in_frame] = BAD;
        }
    }

    if (lost_mappoint_ids.empty()) {
        return lost_mappoint_ids;
    }

    int current_id = points_in_optimization + 2;
    absl::flat_hash_map<ID, int> lost_mappoint_id_to_index;

    for (ID lost_mappoint_id : lost_mappoint_ids_ordered) {
        LandmarkVertex* deformation_vertex = new LandmarkVertex();
        deformation_vertex->setId(current_id);
        deformation_vertex->setToOrigin();

        lost_mappoint_id_to_index[lost_mappoint_id] = current_id;

        current_id++;

        optimizer.addVertex(deformation_vertex);

        // Add regularization edges.
        auto regularization_edges =
                regularization_graph->GetEdges(lost_mappoint_id);

        int n_regularizers = 0;
        for(const auto& [mappoint_id_other, regularization_edge] : regularization_edges) {
            if (n_regularizers > 10) {
                break;
            }

            if (!mappoint_id_to_index.contains(mappoint_id_other)) {
                continue;
            }

            const int idx_other = mappoint_id_to_index[mappoint_id_other];

            float weight = regularization_edge->weight;

            SpatialRegularizerFixed* spatial_regularizer = new SpatialRegularizerFixed();
            spatial_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(current_id - 1)));

            Eigen::Matrix3d spatial_information_matrix = Eigen::Matrix3d::Identity() * info_spatial;
            spatial_regularizer->setInformation(spatial_information_matrix);

            spatial_regularizer->flow_fixed = deformation_vertices[idx_other];
            spatial_regularizer->id1 = lost_mappoint_id;
            spatial_regularizer->id2 = mappoint_id_other;

            g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(th_huber_3dof);
            spatial_regularizer->setRobustKernel(robust_kernel);

            spatial_regularizer->weight_ = weight;

            optimizer.addEdge(spatial_regularizer);

            n_regularizers++;
        }
    }

    camera_pose_vertex->setFixed(true);

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    lost_mappoint_ids.clear();
    for (const auto& [mappoint_id, idx] : lost_mappoint_id_to_index) {
        auto mappoint = map->GetMapPoint(mappoint_id);

        LandmarkVertex* deformation_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx));
        Eigen::Vector3f deformation = deformation_vertex->estimate().cast<float>();

        Eigen::Vector3f previous_landmark_position = mappoint->GetLastWorldPosition();
        Eigen::Vector3f current_landmark_position = deformation + previous_landmark_position;

        mappoint->SetLastWorldPosition(current_landmark_position);

        lost_mappoint_ids.insert(mappoint_id);
    }

    return lost_mappoint_ids;

}

absl::StatusOr<Eigen::Vector3f> DeformableTriangulation(TemporalBuffer& temporal_buffer,
                                                        int candidate_id,
                                                        std::shared_ptr<CameraModel> calibration,
                                                        const float scale) {
    // Recover feature track.
    auto candidate_track = temporal_buffer.GetFeatureTrack(candidate_id);

    // Get candidate neighbors.
    auto neighbour_ids = temporal_buffer.GetClosestMapPointsToFeature(candidate_id, 10, 20, 500);

    if (neighbour_ids.empty()) {
        return absl::InternalError("Feature too close to other ones.");
    }

    int last_frame_id = candidate_track.back().first;
    int first_frame_id = candidate_track.front().first;

    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);

    const float sigma_reprojection = 0.5;   // pixels.
    const float info_reprojection = 1.0f / (sigma_reprojection * sigma_reprojection);

    // Set up vertices.
    absl::flat_hash_map<int, LandmarkVertex*> landmark_vertices;
    absl::flat_hash_map<int, Eigen::Vector3d> landmark_seeds;
    absl::flat_hash_map<int, ReprojectionErrorOnlyDeformation*> frame_id_to_reprojection_edge;

    const auto& [current_frame_id, current_keypoint] = candidate_track.front();
    const auto& [previous_frame_id, previous_keypoint] = candidate_track.back();

    // Unproject rays.
    Eigen::Vector3f current_ray =
            calibration->Unproject(current_keypoint.pt.x, current_keypoint.pt.y).normalized();
    Eigen::Vector3f previous_ray =
            calibration->Unproject(previous_keypoint.pt.x, previous_keypoint.pt.y).normalized();

    // Get camera poses.
    auto current_camera_transform_world = temporal_buffer.GetCameraTransformWorld(current_frame_id);
    auto previous_camera_transform_world = temporal_buffer.GetCameraTransformWorld(previous_frame_id);

    // Rigid triangulation.
    auto landmark_position_status =
            TriangulateMidPoint(previous_ray, current_ray,
                                *previous_camera_transform_world, *current_camera_transform_world);

    if (!landmark_position_status.ok()) {
        return absl::InternalError(landmark_position_status.status().message());
    }

    Eigen::Vector3f landmark_current_position = (*current_camera_transform_world) * (*landmark_position_status);
    cv::Point2f projected_landmark_1 = calibration->Project(landmark_current_position);

    if(SquaredReprojectionError(current_keypoint.pt, projected_landmark_1) > 5.991){
        return absl::InternalError("High reprojection error at first camera.");
    }

    Eigen::Vector3f landmark_previous_position = (*previous_camera_transform_world) * (*landmark_position_status);
    cv::Point2f projected_landmark_2 = calibration->Project(landmark_previous_position);

    if(SquaredReprojectionError(previous_keypoint.pt, projected_landmark_2) > 5.991){
        return absl::InternalError("High reprojection error at second camera.");
    }

    Eigen::Vector3f normal_1 = (*landmark_position_status) - (*current_camera_transform_world).inverse().translation();
    Eigen::Vector3f normal_2 = (*landmark_position_status) - (*previous_camera_transform_world).inverse().translation();
    float parallax = RaysParallax(normal_1, normal_2);

    if(parallax < 0.0025 * 5.f){
        return absl::InternalError("Low parallax.");
    }

    for (const auto& [frame_id, keypoint] : candidate_track) {
        auto camera_transform_world = temporal_buffer.GetCameraTransformWorld(frame_id);
        CHECK_OK(camera_transform_world);

        float depth_seed = 0.0f;
        int n_neighbors = 0;
        for (const auto neighbor_id : neighbour_ids) {
            auto landmark_position = temporal_buffer.GetLandmarkPosition(frame_id, neighbor_id);

            if(landmark_position.ok()) {
                depth_seed += ((*camera_transform_world) * (*landmark_position)).z();

                n_neighbors++;
            }
        }

        if (n_neighbors == 0) {
            return absl::InternalError("Found no neighbours in a temporal point.");
        }

        depth_seed /= (float) n_neighbors;

        if (depth_seed < 0) {
            return absl::InternalError("Negative initial depth.");
        }

        Eigen::Vector3d landmark_position_seed = (calibration->Unproject(keypoint.pt) * depth_seed)
                .cast<double>();

        LandmarkVertex* landmark_vertex = new LandmarkVertex();
        landmark_vertex->setId(frame_id);
        landmark_vertex->setEstimate(landmark_position_seed);

        optimizer.addVertex(landmark_vertex);
        landmark_vertices[frame_id] = landmark_vertex;
        landmark_seeds[frame_id] = landmark_position_seed;

        // Set reprojection error.
        ReprojectionErrorOnlyDeformation* reprojection_error =
                new ReprojectionErrorOnlyDeformation();

        cv::Point2f pixel_coordinates = keypoint.pt;
        Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);
        reprojection_error->setMeasurement(observation);

        reprojection_error->setVertex(0, optimizer.vertex(frame_id));

        Eigen::Matrix2d informationMatrix = Eigen::Matrix2d::Identity() * info_reprojection;
        reprojection_error->setInformation(informationMatrix);

        reprojection_error->calibration_ = calibration;

        optimizer.addEdge(reprojection_error);
        frame_id_to_reprojection_edge[frame_id] = reprojection_error;
    }

    const float th_huber_3dof_squared = 7.815;
    const float th_huber_3dof = sqrt(th_huber_3dof_squared);

    // Set up regularization edges.
    const float sigma_spatial = 0.1;
    const float info_spatial = 1.0f / (sigma_spatial * sigma_spatial);

    vector<SpatialRegularizerWithObservation*> regularization_terms;
    for (auto current_iterator = candidate_track.begin();
         current_iterator != candidate_track.end(); current_iterator++) {
        int current_frame_id = current_iterator->first;

        auto current_camera_transform_world = temporal_buffer.GetCameraTransformWorld(current_frame_id);
        CHECK_OK(current_camera_transform_world);

        g2o::SE3Quat current_pose_g2o = g2o::SE3Quat(
                (*current_camera_transform_world).inverse().unit_quaternion().cast<double>(),
                (*current_camera_transform_world).inverse().translation().cast<double>());

        for (auto next_iterator = next(current_iterator);
             next_iterator != candidate_track.end(); next_iterator++) {
            int next_frame_id = next_iterator->first;

            auto next_camera_transform_world = temporal_buffer.GetCameraTransformWorld(next_frame_id);
            CHECK_OK(next_camera_transform_world);

            g2o::SE3Quat next_pose_g2o = g2o::SE3Quat(
                    (*next_camera_transform_world).inverse().unit_quaternion().cast<double>(),
                    (*next_camera_transform_world).inverse().translation().cast<double>());

            for (auto neighbor_id : neighbour_ids) {
                auto current_landmark_position = temporal_buffer.GetLandmarkPosition(current_frame_id,
                                                                                     neighbor_id);
                auto next_landmark_position = temporal_buffer.GetLandmarkPosition(next_frame_id,
                                                                                  neighbor_id);

                auto first_landmark_position = temporal_buffer.GetLandmarkPosition(first_frame_id,
                                                                                  neighbor_id);

                if (!current_landmark_position.ok() || !next_landmark_position.ok() ||
                    !first_landmark_position.ok()) {
                    continue;
                }

                Eigen::Vector3f flow = (*next_landmark_position) - (*current_landmark_position);

                SpatialRegularizerWithObservation* spatial_regularizer =
                        new SpatialRegularizerWithObservation();

                spatial_regularizer->setMeasurement(flow.cast<double>());

                spatial_regularizer->setVertex(0, optimizer.vertex(current_frame_id));
                spatial_regularizer->setVertex(1, optimizer.vertex(next_frame_id));

                Eigen::Matrix3d informationMatrix = Eigen::Matrix3d::Identity() * info_spatial;
                spatial_regularizer->setInformation(informationMatrix);

                spatial_regularizer->weight_ = 1.0f;

                spatial_regularizer->next_world_transform_camera_ = next_pose_g2o;
                spatial_regularizer->current_world_transform_camera_ = current_pose_g2o;

                optimizer.addEdge(spatial_regularizer);

                regularization_terms.push_back(spatial_regularizer);
            }
        }
    }

    if(optimizer.edges().size() == 0 || optimizer.vertices().size() == 0) {
        return absl::InternalError("Optimization is empty.");
    }

    // optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Remove outlier regularization terms.
    int bad_edges = 0;
    for (auto regularization_edge : regularization_terms) {
        regularization_edge->computeError();

        regularization_edge->setRobustKernel(nullptr);

        if (regularization_edge->chi2() > th_huber_3dof_squared) {
            regularization_edge->setLevel(1);
            bad_edges++;
        }
    }

    if ((float)bad_edges / (float)regularization_terms.size() > 0.5) {
        return absl::InternalError("Triangulation has to many bad neighbors.");
    }

    int n_bad_edges = 0;

    for (const auto [id, reprojection_edge] : frame_id_to_reprojection_edge) {
        reprojection_edge->computeError();
        if (reprojection_edge->chi2() > 5.99 * 10) {
            n_bad_edges++;
        }
    }

    if ((float) n_bad_edges / (float) optimizer.vertices().size() > 0.5) {
        return absl::InternalError("Triangulation has to much error.");
    }

    cv::Point2f latest_pixel_coordinates = candidate_track.back().second.pt;
    float current_depth = landmark_vertices[last_frame_id]->estimate().z();

    auto last_camera_transform_world = temporal_buffer.GetCameraTransformWorld(last_frame_id);
    CHECK_OK(last_camera_transform_world);

    Eigen::Vector3f unprojected_keypoint = calibration->Unproject(latest_pixel_coordinates);
    unprojected_keypoint /= unprojected_keypoint.z();

    Eigen::Vector3f triangulated_landmark_position =
            ((*last_camera_transform_world).inverse() * (unprojected_keypoint * current_depth));

    return triangulated_landmark_position;
}

class TemporalPoint {
public:
    TemporalPoint() = delete;

    TemporalPoint(ID mappoint_id_1, ID mappoint_id_2, ID keyframe_id_1, ID keyframe_id_2){
        mappoint_id_1_ = min(mappoint_id_1, mappoint_id_2);
        mappoint_id_2_ = max(mappoint_id_1, mappoint_id_2);
        keyframe_id_1_ = min(keyframe_id_1, keyframe_id_2);
        keyframe_id_2_ = max(keyframe_id_1, keyframe_id_2);
    }

    bool operator==(const TemporalPoint& rhs) const {
        return this->mappoint_id_1_ == rhs.mappoint_id_1_ && this->mappoint_id_2_ == rhs.mappoint_id_2_ &&
               this->keyframe_id_1_ == rhs.keyframe_id_1_ && this->keyframe_id_2_ == rhs.keyframe_id_2_;
    }

    struct HashFunction
    {
        size_t operator()(const TemporalPoint& point) const
        {
            size_t mappoint_1_hash = std::hash<ID>()(point.mappoint_id_1_);
            size_t mappoint_2_hash = std::hash<ID>()(point.keyframe_id_2_) << 1;
            size_t keyframe_1_hash = std::hash<ID>()(point.keyframe_id_1_) << 2;
            size_t keyframe_2_hash = std::hash<ID>()(point.keyframe_id_2_) << 3;
            return mappoint_1_hash ^ mappoint_2_hash ^ keyframe_1_hash ^ keyframe_2_hash;
        }
    };

private:
    ID mappoint_id_1_, mappoint_id_2_;
    ID keyframe_id_1_, keyframe_id_2_;
};

class SpatialPoint {
public:
    SpatialPoint() = delete;

    SpatialPoint(ID mappoint_id_1, ID mappoint_id_2, ID keyframe_id){
        mappoint_id_1_ = min(mappoint_id_1, mappoint_id_2);
        mappoint_id_2_ = max(mappoint_id_1, mappoint_id_2);
        keyframe_id_ = keyframe_id;
    }

    bool operator==(const SpatialPoint& rhs) const {
        return this->mappoint_id_1_ == rhs.mappoint_id_1_ && this->mappoint_id_2_ == rhs.mappoint_id_2_ &&
               this->keyframe_id_ == rhs.keyframe_id_;
    }

    struct HashFunction
    {
        size_t operator()(const SpatialPoint& point) const
        {
            size_t mappoint_1_hash = std::hash<ID>()(point.mappoint_id_1_);
            size_t mappoint_2_hash = std::hash<ID>()(point.mappoint_id_2_) << 1;
            size_t keyframe_hash = std::hash<ID>()(point.keyframe_id_) << 2;
            return mappoint_1_hash ^ mappoint_2_hash ^ keyframe_hash;
        }
    };

private:
    ID mappoint_id_1_, mappoint_id_2_;
    ID keyframe_id_;
};

void LocalDeformableBundleAdjustment(std::shared_ptr<Map> map,
                                     const float scale) {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);

    auto keyframes = map->GetKeyFrames();

    const int max_keyframes_in_optimization = 5;
    vector<shared_ptr<KeyFrame>> keyframes_in_optimization;

    // Get window of KeyFrames and set their vertices
    int biggest_keyframe_idx = 0;
    for (auto it = keyframes.rbegin(); it != keyframes.rend(); it++) {
        if (keyframes_in_optimization.size() >= max_keyframes_in_optimization) {
            break;
        }

        auto keyframe = it->second;

        g2o::VertexSE3Expmap* camera_pose_vertex = new g2o::VertexSE3Expmap();
        Sophus::SE3f camera_transform_world = keyframe->CameraTransformationWorld();
        camera_pose_vertex->setEstimate(g2o::SE3Quat(
                camera_transform_world.unit_quaternion().cast<double>(),
                camera_transform_world.translation().cast<double>()));
        camera_pose_vertex->setId(keyframe->GetId());

        optimizer.addVertex(camera_pose_vertex);

        keyframes_in_optimization.push_back(keyframe);

        if (biggest_keyframe_idx < keyframe->GetId()) {
            biggest_keyframe_idx = keyframe->GetId();
        }
    }

    if (keyframes_in_optimization.size() < 3) {
        return;
    }

    // Set landmark vertices.
    absl::btree_map<ID, absl::btree_map<ID, int>> inserted_landmarks;
    int current_optimization_idx = biggest_keyframe_idx + 1;
    int n_inserted_landmarks = 0;
    for (auto it = keyframes_in_optimization.rbegin(); it != keyframes_in_optimization.rend(); it++) {
        auto keyframe = *it;
        ID keyframe_id = keyframe->GetId();

        auto landmark_positions = keyframe->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
        auto mappoint_ids = keyframe->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

        for (int idx = 0; idx < landmark_positions.size(); idx++) {
            ID mappoint_id = mappoint_ids[idx];
            Eigen::Vector3f landmark_position = landmark_positions[idx];

            auto landmark_vertex = new LandmarkVertex();
            landmark_vertex->setId(current_optimization_idx);
            landmark_vertex->setEstimate(landmark_position.cast<double>());

            optimizer.addVertex(landmark_vertex);

            inserted_landmarks[keyframe_id][mappoint_id] = current_optimization_idx;

            current_optimization_idx++;
            n_inserted_landmarks++;
        }
    }

    LOG(INFO) << "Optimization summary:";
    LOG(INFO) << "\t-Number of KeyFrames: " << keyframes_in_optimization.size();
    LOG(INFO) << "\t-Number of landmarks: " << n_inserted_landmarks;

    const int regularizers_per_point = 10;

    const float th_huber_2dof_squared = 5.99;
    const float th_huber_2dof = sqrt(th_huber_2dof_squared);

    const float th_huber_3dof_squared = 0.584;
    const float th_huber_3dof = sqrt(th_huber_3dof_squared);

    const float sigma_reprojection = 0.5;   // pixels.
    const float info_reprojection = 1.0f / (sigma_reprojection * sigma_reprojection);

    const float sigma_position = 0.1f;//  * 0.0394105;
    const float info_position = 1.0f / (sigma_position * sigma_position);

    const float sigma_spatial = 0.1 * scale;   // mm.
    const float info_spatial = 1.0f / (sigma_spatial * sigma_spatial);

    auto regularization_graph = map->GetRegularizationGraph();

    // Point vertices will be added as we add observations. This map stores the landmarks vertex index
    // in the optimization in the following manner:
    //      inserted_landmarks[keyframe_id][mappoint_id] = idx_in_optimization;
    absl::flat_hash_set<SpatialPoint, SpatialPoint::HashFunction> spring_edges;
    absl::flat_hash_set<TemporalPoint,TemporalPoint::HashFunction> dumper_edges;
    for (auto it = keyframes_in_optimization.rbegin(); it != keyframes_in_optimization.rend(); it++) {
        auto keyframe = *it;
        ID keyframe_id = keyframe->GetId();

        auto next_it = next(it);
        shared_ptr<KeyFrame> next_keyframe = nullptr;
        int next_keyframe_id = -1;
        if (next_it != keyframes_in_optimization.rend()) {
            next_keyframe = *next_it;
            next_keyframe_id = next_keyframe->GetId();
        }

        auto landmark_positions = keyframe->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
        auto mappoint_ids = keyframe->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
        auto keypoints = keyframe->GetKeypointsWithStatus({TRACKED_WITH_3D});

        for (int idx = 0; idx < landmark_positions.size(); idx++) {
            ID mappoint_id = mappoint_ids[idx];
            Eigen::Vector3f landmark_position = landmark_positions[idx];
            cv::KeyPoint keypoint = keypoints[idx];

            int landmark_optimization_index = inserted_landmarks[keyframe_id][mappoint_id];

            // Set reprojection error.
            auto reprojection_error = new ReprojectionError();

            cv::Point2f pixel_coordinates = keypoint.pt;
            Eigen::Matrix<double, 2, 1> observation(pixel_coordinates.x, pixel_coordinates.y);

            reprojection_error->setMeasurement(observation);

            reprojection_error->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                    optimizer.vertex(keyframe_id)));
            reprojection_error->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                    optimizer.vertex(landmark_optimization_index)));

            reprojection_error->setInformation(Eigen::Matrix2d::Identity() * info_reprojection);

            g2o::RobustKernelHuber *robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(th_huber_2dof);
            reprojection_error->setRobustKernel(robust_kernel);

            reprojection_error->calibration_ = keyframe->GetCalibration();

            optimizer.addEdge(reprojection_error);

            // Set up position regularizers.
            auto regularization_edges =
                    regularization_graph->GetEdges(mappoint_id);

            int n_regularizers = 0;
            for (const auto &[mappoint_id_other, regularization_edge]: regularization_edges) {
                // Check if there is already enough regularizers or if the connection is good.
                if (n_regularizers > regularizers_per_point ||
                    regularization_edge->status == RegularizationGraph::BAD) {
                    break;
                }

                // Check the other landmark is observed by the Keyframe.
                if (!inserted_landmarks[keyframe_id].contains(mappoint_id_other)) {
                    continue;
                }

                // Check if this regularizer has already been inserted in the optimization.
                SpatialPoint spring_connection(mappoint_id, mappoint_id_other, keyframe_id);
                if (spring_edges.contains(spring_connection)) {
                    n_regularizers++;
                    continue;
                }

                spring_edges.insert(spring_connection);

                const int other_landmark_optimization_index =
                        inserted_landmarks[keyframe_id][mappoint_id_other];

                auto position_regularizer = new PositionRegularizer();
                position_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(landmark_optimization_index)));
                position_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(other_landmark_optimization_index)));

                position_regularizer->setMeasurement(regularization_edge->first_distance);

                Eigen::Matrix<double, 1, 1> position_information_matrix =
                        Eigen::Matrix<double, 1, 1>::Identity() * info_position;
                position_regularizer->setInformation(position_information_matrix);

                position_regularizer->k_ = 1.1f;

                optimizer.addEdge(position_regularizer);

                n_regularizers++;
            }

            if (next_keyframe) {
                if (!inserted_landmarks[next_keyframe_id].contains(mappoint_id)) {
                    continue;
                }

                const int next_landmark_optimization_index =
                        inserted_landmarks[next_keyframe_id][mappoint_id];

                // Set up position regularizers
                int n_regularizers = 0;
                for (const auto &[mappoint_id_other, regularization_edge]: regularization_edges) {
                    // Check if there is already enough regularizers or if the connection is good.
                    if (n_regularizers > regularizers_per_point ||
                        regularization_edge->status == RegularizationGraph::BAD) {
                        break;
                    }

                    // Check both Mappoints are observed by the next Keyframe
                    if (!inserted_landmarks[keyframe_id].contains(mappoint_id_other) ||
                        !inserted_landmarks[next_keyframe_id].contains(mappoint_id_other)) {
                        continue;
                    }

                    // Check if this regularizer has already been inserted in the optimization.
                    TemporalPoint dumper_connection(mappoint_id, mappoint_id_other, keyframe_id, next_keyframe_id);
                    if (dumper_edges.contains(dumper_connection)) {
                        n_regularizers++;
                        continue;
                    }

                    dumper_edges.insert(dumper_connection);

                    const int other_landmark_optimization_index =
                            inserted_landmarks[keyframe_id][mappoint_id_other];
                    const int next_other_landmark_optimization_index =
                            inserted_landmarks[next_keyframe_id][mappoint_id_other];

                    SpatialRegularizer* spatial_regularizer = new SpatialRegularizer();
                    spatial_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(landmark_optimization_index)));
                    spatial_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(other_landmark_optimization_index)));
                    spatial_regularizer->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(next_landmark_optimization_index)));
                    spatial_regularizer->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(next_other_landmark_optimization_index)));

                    Eigen::Matrix3d spatial_information_matrix = Eigen::Matrix3d::Identity() * info_spatial;
                    spatial_regularizer->setInformation(spatial_information_matrix);

                    g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
                    robust_kernel->setDelta(th_huber_3dof);
                    spatial_regularizer->setRobustKernel(robust_kernel);

                    spatial_regularizer->weight_ = regularization_edge->weight;

                    optimizer.addEdge(spatial_regularizer);

                    n_regularizers++;
                }
            }
        }
    }

    // Perform optimization.
    optimizer.setVerbose(false);
    optimizer.initializeOptimization(0);
    optimizer.optimize(5);

    // Recover optimized variables.
    for (const auto [keyframe_id, mappoint_id_to_idx] : inserted_landmarks) {
        // Update KeyFrame pose
        auto keyframe_vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(keyframe_id));
        auto keyframe = map->GetKeyFrame(keyframe_id);
        keyframe->CameraTransformationWorld() = Sophus::SE3f(
                keyframe_vertex->estimate().to_homogeneous_matrix().cast<float>());

        // Update landmark positions
        for (const auto [mappoint_id, idx_in_optimization] : mappoint_id_to_idx) {
            auto landmark_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx_in_optimization));

            int idx_in_keyframe = keyframe->MapPointIdToIndex().at(mappoint_id);
            keyframe->LandmarkPositions()[idx_in_keyframe] = landmark_vertex->estimate().cast<float>();
        }
    }
}
