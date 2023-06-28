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

#include "keyframe.h"

using namespace std;

long unsigned int KeyFrame::nextId_ = 0;

KeyFrame::KeyFrame(Frame &frame) {
    // Recover tracked landmarks with 3D.
    keypoints_ = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    landmark_positions_ = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    landmark_ground_truth_ = frame.GetGroundTruthWithStatus({TRACKED_WITH_3D});

    landmark_status_.resize(keypoints_.size());
    fill(landmark_status_.begin(), landmark_status_.end(), TRACKED_WITH_3D);

    vector<ID> mappoint_ids = frame.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
    mappoint_id_to_index_.clear();
    index_to_mappoint_id_.clear();
    for(int idx = 0; idx < mappoint_ids.size(); idx++) {
        mappoint_id_to_index_[mappoint_ids[idx]] = idx;
        index_to_mappoint_id_[idx] = mappoint_ids[idx];
    }

    // Recover tracked landmakrs without 3D.
    vector<cv::KeyPoint> keypoints_without_3d = frame.GetKeypointsWithStatus({TRACKED});
    keypoints_.insert(keypoints_.end(), keypoints_without_3d.begin(), keypoints_without_3d.end());

    landmark_positions_.resize(keypoints_.size(), Eigen::Vector3f::Zero());
    landmark_status_.resize(keypoints_.size(), TRACKED);
    landmark_ground_truth_.resize(keypoints_.size(), absl::InternalError("No ground truth available."));

    camera_transformation_world_ = frame.CameraTransformationWorld();
    calibration_ = frame.GetCalibration();

    id_ = nextId_++;
}

std::vector<cv::KeyPoint> &KeyFrame::Keypoints() {
    return keypoints_;
}

std::vector <cv::KeyPoint> KeyFrame::GetKeypointsWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<cv::KeyPoint> keypoints;
    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            keypoints.push_back(keypoints_[idx]);
        }
    }

    return keypoints;
}

std::vector<Eigen::Vector3f> &KeyFrame::LandmarkPositions() {
    return landmark_positions_;
}

std::vector <Eigen::Vector3f> KeyFrame::GetLandmarkPositionsWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<Eigen::Vector3f> landmark_positions;
    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            landmark_positions.push_back(landmark_positions_[idx]);
        }
    }

    return landmark_positions;
}

std::vector<LandmarkStatus>& KeyFrame::LandmarkStatuses() {
    return landmark_status_;
}

Sophus::SE3f& KeyFrame::CameraTransformationWorld() {
    return camera_transformation_world_;
}

const absl::flat_hash_map<int, ID> &KeyFrame::IndexToMapPointId() const {
    return index_to_mappoint_id_;
}

const absl::flat_hash_map<ID, int> &KeyFrame::MapPointIdToIndex() const {
    return mappoint_id_to_index_;
}

std::shared_ptr<CameraModel> KeyFrame::GetCalibration() {
    return calibration_;
}

std::vector<ID> KeyFrame::GetMapPointsIdsWithStatus(
        const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<ID> mappoints_ids;

    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx]) && index_to_mappoint_id_.contains(idx)) {
            mappoints_ids.push_back(index_to_mappoint_id_[idx]);
        }
    }

    return mappoints_ids;
}

long unsigned int KeyFrame::GetId() {
    return id_;
}

std::vector<absl::StatusOr<Eigen::Vector3f>> KeyFrame::GroundTruth() {
    return landmark_ground_truth_;
}

std::vector<absl::StatusOr<Eigen::Vector3f>>
KeyFrame::GetGroundTruthWithStatus(const absl::flat_hash_set<LandmarkStatus> statuses) {
    vector<absl::StatusOr<Eigen::Vector3f>> ground_truth;

    for(int idx = 0; idx < landmark_status_.size(); idx++) {
        if (statuses.contains(landmark_status_[idx])) {
            ground_truth.push_back(landmark_ground_truth_[idx]);
        }
    }

    return ground_truth;
}
