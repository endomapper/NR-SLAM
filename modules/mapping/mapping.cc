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

#include <fstream>
#include "mapping.h"

#include "optimization/g2o_optimization.h"
#include "utilities/geometry_toolbox.h"

#include "absl/log/log.h"
#include "absl/log/check.h"

using namespace std;

Mapping::Mapping(std::shared_ptr<Map> map, std::shared_ptr<CameraModel> calibration,
                 const Options options, TimeProfiler* time_profiler) :
        options_(options), map_(map), calibration_(calibration), time_profiler_(time_profiler) {
}

void Mapping::DoMapping() {
    if (map_->IsEmpty()) {
        return;
    }

    // Get next KeyFrame to process
    auto keyframe = map_->GetNextUnmappedKeyFrame();

    if (true && keyframe) {
        // If the KeyFrame is valid, do KeyFrame mapping.
        KeyFrameMapping();

        // Update tracking frame with the optimized KeyFrame.
        UpdateTrackingFrameFromKeyFrame(keyframe);
    } else {
        // If there is no KeyFrame to process, do Frame mapping.
        FrameMapping();
    }
}

void Mapping::KeyFrameMapping() {
    LocalDeformableBundleAdjustment(map_, map_->GetMapScale());
}

void Mapping::FrameMapping() {
    // Triangulate new landmarks.
    LandmarkTriangulation();
}

void Mapping::LandmarkTriangulation() {
    auto temporal_buffer = map_->GetTemporalBuffer();

    // Iterate over the KeyPoints tracked in the last frame
    vector<int> triangulation_candidate_ids = temporal_buffer->GetTriangulationCandidatesIds();

    auto current_frame = map_->GetMutableLastFrame();

    auto calibration = current_frame->GetCalibration();

    int triangulated_landmarks = 0;

    const int current_frame_id = current_frame->GetId() - 1;

    int n_rigid_triangulations = 0;
    int n_deformable_triangulations = 0;

    vector<absl::StatusOr<Eigen::Vector3f>> rigid_triangulations;
    vector<absl::StatusOr<Eigen::Vector3f>> deformable_triangulations;

    // Try to triangulate candidates.
    vector<int> candidates_triangulated;
    vector<ID> landmark_ids_triangulated;
    for (auto candidate_id : triangulation_candidate_ids) {
        // Check there are no close features.
        auto neighbour_ids = temporal_buffer->GetClosestMapPointsToFeature(candidate_id, 10, 20, 500);
        if (neighbour_ids.empty()) {
            rigid_triangulations.push_back(absl::InternalError("Close features"));
            deformable_triangulations.push_back(absl::InternalError("Close features"));
            continue;
        }

        if (temporal_buffer->TrackLenght(candidate_id) >= 5) {
            auto landmark_triangulated = DeformableLandmarkTriangulation(candidate_id);

            if (landmark_triangulated.ok()) {
                if ((*landmark_triangulated).hasNaN()) {
                    deformable_triangulations.push_back(absl::InternalError("NaN."));
                } else {
                    deformable_triangulations.push_back(landmark_triangulated);
                    if (landmark_triangulated.ok()) {
                        n_deformable_triangulations++;
                    }
                }
            } else {
                deformable_triangulations.push_back(landmark_triangulated);
            }

        } else {
            deformable_triangulations.push_back(absl::InternalError("Short track"));
        }

        // Recover feature track.
        auto candidate_track = temporal_buffer->GetFeatureTrack(candidate_id);

        const auto& [current_frame_id, current_keypoint] = candidate_track.front();
        const auto& [previous_frame_id, previous_keypoint] = candidate_track.back();

        // Rigidity condition.
        if (!temporal_buffer->CheckRigidity(current_frame_id, previous_frame_id, 0.004)) {
            rigid_triangulations.push_back(absl::InternalError("Rigidity not detected"));
            continue;
        }

        // Unproject rays.
        Eigen::Vector3f current_ray =
                calibration->Unproject(current_keypoint.pt.x, current_keypoint.pt.y).normalized();
        Eigen::Vector3f previous_ray =
                calibration->Unproject(previous_keypoint.pt.x, previous_keypoint.pt.y).normalized();

        // Get camera poses.
        auto current_camera_transform_world = temporal_buffer->GetCameraTransformWorld(current_frame_id);
        auto previous_camera_transform_world = temporal_buffer->GetCameraTransformWorld(previous_frame_id);

        auto landmark_position_status =
                TriangulateMidPoint(previous_ray, current_ray,
                                    *previous_camera_transform_world, *current_camera_transform_world);

        if (!landmark_position_status.ok()) {
            rigid_triangulations.push_back(landmark_position_status);
            continue;
        }

        Eigen::Vector3f normal_1 = (*landmark_position_status) - (*current_camera_transform_world).inverse().translation();
        Eigen::Vector3f normal_2 = (*landmark_position_status) - (*previous_camera_transform_world).inverse().translation();
        float parallax = RaysParallax(normal_1, normal_2);

        if(parallax < options_.rad_per_pixel * 10.f || parallax > options_.rad_per_pixel * 20.f){
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        // Check Reprojection error.
        Eigen::Vector3f landmark_position_1 = (*previous_camera_transform_world) * (*landmark_position_status);

        if (landmark_position_1.z() < 0) {
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        cv::Point2f projected_landmark_1 = calibration->Project(landmark_position_1);

        if(SquaredReprojectionError(previous_keypoint.pt, projected_landmark_1) > 5.991){
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        Eigen::Vector3f landmark_position_2 = (*current_camera_transform_world) * (*landmark_position_status);

        if (landmark_position_2.z() < 0) {
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        cv::Point2f projected_landmark_2 = calibration->Project(landmark_position_2);

        if(SquaredReprojectionError(current_keypoint.pt, projected_landmark_2) > 5.991){
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        rigid_triangulations.push_back(landmark_position_status);
        if (landmark_position_status.ok()) {
            n_rigid_triangulations++;
        }
    }

    for (int idx = 0; idx < triangulation_candidate_ids.size(); idx++) {
        auto candidate_id = triangulation_candidate_ids[idx];
        Eigen::Vector3f landmark_triangulated;
        if (n_rigid_triangulations > 1.5 * n_deformable_triangulations) {
            if (!rigid_triangulations[idx].ok()) {
                continue;
            } else {
                landmark_triangulated = *(rigid_triangulations[idx]);
            }
        } else if (n_deformable_triangulations >= 1.5 * n_rigid_triangulations){
            if (!deformable_triangulations[idx].ok()) {
                continue;
            } else {
                landmark_triangulated = *deformable_triangulations[idx];
            }
        } else {
            continue;
        }

        const auto index_in_frame = temporal_buffer->GetLandmarkIndexInFrame(current_frame_id,
                                                                             candidate_id);

        if(landmark_triangulated.hasNaN()) {
            LOG(INFO) << landmark_triangulated.transpose();
            continue;
        }

        if (!index_in_frame.ok()) {
            LOG(INFO) << index_in_frame.status().message();
            continue;
        }

        // Create MapPoint and insert it into the map.
        auto mappoint = map_->CreateAndInsertMapPoint(landmark_triangulated, candidate_id);

        // Insert it into the tracking.
        current_frame->AddGeometryToKeypoint(*index_in_frame, landmark_triangulated,
                                             mappoint->GetId());

        candidates_triangulated.push_back(candidate_id);

        landmark_ids_triangulated.push_back(mappoint->GetId());

        triangulated_landmarks++;
    }

    // Add newly triangulated landmarks to the regularization graph.
    auto current_mappoints_ids = current_frame->GetMapPointsIdsWithStatus({TRACKED_WITH_3D, JUST_TRIANGULATED});
    for (auto landmark_id : landmark_ids_triangulated) {
        auto landmark_position_status = current_frame->LandmarkPosition(landmark_id);
        CHECK_OK(landmark_position_status);

        for (auto other_landmark_id : current_mappoints_ids) {
            if (landmark_id == other_landmark_id) {
                continue;
            }

            auto other_landmark_position_status = current_frame->LandmarkPosition(other_landmark_id);
            CHECK_OK(other_landmark_position_status);

            Eigen::Vector3f relative_position = *other_landmark_position_status - *landmark_position_status;

            map_->GetRegularizationGraph()->AddEdge(landmark_id, other_landmark_id, relative_position);
        }
    }
}

absl::StatusOr<Eigen::Vector3f> Mapping::DeformableLandmarkTriangulation(const int candidate_id) {
    return DeformableTriangulation(*map_->GetTemporalBuffer(),
                                                         candidate_id,
                                                         calibration_,
                                                         1.f);
}

void Mapping::UpdateTrackingFrameFromKeyFrame(std::shared_ptr<KeyFrame> keyframe) {
    auto current_frame = map_->GetMutableLastFrame();

    current_frame->SetFromKeyFrame(keyframe);
}
