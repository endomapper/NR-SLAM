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

#include "frame.h"

#include "absl/log/log.h"
#include "absl/log/check.h"

using namespace std;

Frame::Frame() {}

Frame::Frame(const Frame &other) {
    keypoints_ = other.keypoints_;
    landmark_positions_ = other.landmark_positions_;
    landmark_status_ = other.landmark_status_;
    landmark_ground_truth_ = other.landmark_ground_truth_;

    CHECK_EQ(landmark_positions_.size(), landmark_ground_truth_.size());

    mappoint_id_to_index_ = other.mappoint_id_to_index_;
    index_to_mappoint_id_ = other.index_to_mappoint_id_;

    camera_transformation_world_ = other.camera_transformation_world_;

    calibration_ = calibration_;

    id_ = other.id_;
}

void Frame::SetFromKeyFrame(std::shared_ptr<KeyFrame> keyframe) {
    // Recover tracked landmarks with 3D.
    keypoints_ = keyframe->GetKeypointsWithStatus({TRACKED_WITH_3D});
    landmark_positions_ = keyframe->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    landmark_ground_truth_ = keyframe->GetGroundTruthWithStatus({TRACKED_WITH_3D});

    landmark_status_.resize(keypoints_.size());
    fill(landmark_status_.begin(), landmark_status_.end(), TRACKED_WITH_3D);

    vector<ID> mappoint_ids = keyframe->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
    mappoint_id_to_index_.clear();
    index_to_mappoint_id_.clear();
    for(int idx = 0; idx < mappoint_ids.size(); idx++) {
        mappoint_id_to_index_[mappoint_ids[idx]] = idx;
        index_to_mappoint_id_[idx] = mappoint_ids[idx];
    }

    // Recover tracked landmarks without 3D.
    vector<cv::KeyPoint> keypoints_without_3d = keyframe->GetKeypointsWithStatus({TRACKED});
    keypoints_.insert(keypoints_.end(), keypoints_without_3d.begin(), keypoints_without_3d.end());

    landmark_positions_.resize(keypoints_.size(), Eigen::Vector3f::Zero());
    landmark_status_.resize(keypoints_.size(), TRACKED);
    landmark_ground_truth_.resize(keypoints_.size(), absl::InternalError("No ground truth available."));

    // Copy rest of fields.
    camera_transformation_world_ = keyframe->CameraTransformationWorld();
    calibration_ = keyframe->GetCalibration();

    CHECK_EQ(landmark_positions_.size(), landmark_ground_truth_.size());
}

std::vector<cv::KeyPoint> &Frame::Keypoints() {
    return keypoints_;
}

std::vector <cv::KeyPoint> Frame::GetKeypointsWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) const {
    vector<cv::KeyPoint> keypoints;
    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            keypoints.push_back(keypoints_[idx]);
        }
    }

    return keypoints;
}

std::vector<Eigen::Vector3f> &Frame::LandmarkPositions() {
    return landmark_positions_;
}

absl::StatusOr<Eigen::Vector3f> Frame::LandmarkPosition(ID mappoint_id) {
    if (!mappoint_id_to_index_.contains(mappoint_id) ||
        (landmark_status_[mappoint_id_to_index_[mappoint_id]] != TRACKED_WITH_3D &&
        landmark_status_[mappoint_id_to_index_[mappoint_id]] != JUST_TRIANGULATED)) {
        return absl::InternalError("MapPoint is not present in the Frame.");
    }

    return landmark_positions_[mappoint_id_to_index_[mappoint_id]];
}

std::vector <Eigen::Vector3f> Frame::GetLandmarkPositionsWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) const {
    vector<Eigen::Vector3f> landmark_positions;
    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            landmark_positions.push_back(landmark_positions_[idx]);
        }
    }

    return landmark_positions;
}

std::vector<LandmarkStatus>& Frame::LandmarkStatuses() {
    return landmark_status_;
}

void Frame::InsertObservation(const cv::KeyPoint& keypoint, const Eigen::Vector3f& landmark_position,
                              const ID mappoint_id, const LandmarkStatus status) {
    if (status == TRACKED_WITH_3D) {
        mappoint_id_to_index_[mappoint_id] = keypoints_.size();
        index_to_mappoint_id_[keypoints_.size()] = mappoint_id;
    }


    keypoints_.push_back(keypoint);
    landmark_positions_.push_back(landmark_position);
    landmark_status_.push_back(status);
    landmark_ground_truth_.push_back(absl::InternalError("No ground truth available."));
}

void Frame::AddGeometryToKeypoint(const int idx, const Eigen::Vector3f &landmark_position, const ID mappoint_id) {
    CHECK_LT(idx, landmark_positions_.size());
    CHECK(landmark_status_[idx] != JUST_TRIANGULATED && landmark_status_[idx] != TRACKED_WITH_3D);

    landmark_positions_[idx] = landmark_position;
    landmark_status_[idx] = JUST_TRIANGULATED;
    index_to_mappoint_id_[idx] = mappoint_id;
    mappoint_id_to_index_[mappoint_id] = idx;
}

void Frame::Clear() {
    keypoints_.clear();
    landmark_positions_.clear();
    landmark_status_.clear();

    mappoint_id_to_index_.clear();
    index_to_mappoint_id_.clear();
}

Sophus::SE3f& Frame::MutableCameraTransformationWorld() {
    return camera_transformation_world_;
}

Sophus::SE3f Frame::CameraTransformationWorld() const {
    return camera_transformation_world_;
}

const absl::flat_hash_map<int, ID> &Frame::IndexToMapPointId() const {
    return index_to_mappoint_id_;
}

const absl::flat_hash_map<ID, int> &Frame::MapPointIdToIndex() const {
    return mappoint_id_to_index_;
}

void Frame::SetCalibration(std::shared_ptr<CameraModel> calibration) {
    calibration_ = calibration;
}

std::shared_ptr<CameraModel> Frame::GetCalibration() {
    return calibration_;
}

std::vector<ID> Frame::GetMapPointsIdsWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<ID> mappoints_ids;

    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx]) && index_to_mappoint_id_.contains(idx)) {
            mappoints_ids.push_back(index_to_mappoint_id_[idx]);
        }
    }

    return mappoints_ids;
}

void Frame::IncreaseId() {
    id_++;
}

int Frame::GetId() {
    return id_;
}

std::vector<LandmarkStatus> Frame::GetLandmarkStatusesWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<LandmarkStatus> landmark_statuses;

    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            landmark_statuses.push_back(landmark_status_[idx]);
        }
    }

    return landmark_statuses;
}

std::vector<int> Frame::GetIndexWithStatus(const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<int> indexes;

    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            indexes.push_back(idx);
        }
    }

    return indexes;
}

std::vector<absl::StatusOr<Eigen::Vector3f>>& Frame::MutableGroundTruth() {
    return landmark_ground_truth_;
}

std::vector<absl::StatusOr<Eigen::Vector3f>> Frame::GroundTruth() {
    return landmark_ground_truth_;
}

std::vector<absl::StatusOr<Eigen::Vector3f>>
Frame::GetGroundTruthWithStatus(const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<absl::StatusOr<Eigen::Vector3f>> ground_truth;

    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            ground_truth.push_back(landmark_ground_truth_[idx]);
        }
    }

    return ground_truth;
}

void Frame::SetDeformationMaginitud(const float deformation_magnitud) {
    median_deformation_magnitud_ = deformation_magnitud;
}

float Frame::GetDeformationMagnitud() {
    return median_deformation_magnitud_;
}
