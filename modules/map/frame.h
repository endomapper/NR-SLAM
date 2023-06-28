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

#ifndef NRSLAM_FRAME_H
#define NRSLAM_FRAME_H

#include <vector>

#include "calibration/camera_model.h"
#include "map/keyframe.h"
#include "map/mappoint.h"
#include "utilities/landmark_status.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "sophus/se3.hpp"

#include <opencv2/opencv.hpp>

typedef long unsigned int ID;

class KeyFrame;

class Frame {
public:
    Frame();

    Frame(const Frame& other);

    void SetFromKeyFrame(std::shared_ptr<KeyFrame> keyframe);

    std::vector<cv::KeyPoint>& Keypoints();

    std::vector<cv::KeyPoint> GetKeypointsWithStatus(
            const absl::flat_hash_set<LandmarkStatus> statuses) const;

    std::vector<Eigen::Vector3f>& LandmarkPositions();

    absl::StatusOr<Eigen::Vector3f> LandmarkPosition(ID mappoint_id);

    std::vector<Eigen::Vector3f> GetLandmarkPositionsWithStatus(
            const absl::flat_hash_set<LandmarkStatus> statuses) const;

    std::vector<LandmarkStatus>& LandmarkStatuses();

    std::vector<LandmarkStatus> GetLandmarkStatusesWithStatus(
            const absl::flat_hash_set<LandmarkStatus> statuses);

    std::vector<int> GetIndexWithStatus(
            const absl::flat_hash_set<LandmarkStatus> statuses);

    void InsertObservation(const cv::KeyPoint& keypoint, const Eigen::Vector3f& landmark_position,
                           const ID mappoint_id, const LandmarkStatus status);

    void AddGeometryToKeypoint(const int idx, const Eigen::Vector3f& landmark_position,
                               const ID mappoint_id);

    Sophus::SE3f& MutableCameraTransformationWorld();

    Sophus::SE3f CameraTransformationWorld() const;

    const absl::flat_hash_map<int, ID>& IndexToMapPointId() const;

    const absl::flat_hash_map<ID, int>& MapPointIdToIndex() const;

    void SetCalibration(std::shared_ptr<CameraModel> calibration);

    std::shared_ptr<CameraModel> GetCalibration();

    std::vector<ID> GetMapPointsIdsWithStatus(
            const absl::flat_hash_set<LandmarkStatus> statuses);

    std::vector<absl::StatusOr<Eigen::Vector3f>>& MutableGroundTruth();

    std::vector<absl::StatusOr<Eigen::Vector3f>> GroundTruth();

    std::vector<absl::StatusOr<Eigen::Vector3f>> GetGroundTruthWithStatus(
            const absl::flat_hash_set<LandmarkStatus> statuses);

    void IncreaseId();

    int GetId();

    void Clear();

    void SetDeformationMaginitud(const float deformation_magnitud);

    float GetDeformationMagnitud();

private:
    std::vector<cv::KeyPoint> keypoints_;
    std::vector<Eigen::Vector3f> landmark_positions_;
    std::vector<LandmarkStatus> landmark_status_;

    std::vector<absl::StatusOr<Eigen::Vector3f>> landmark_ground_truth_;

    absl::flat_hash_map<ID, int> mappoint_id_to_index_;
    absl::flat_hash_map<int, ID> index_to_mappoint_id_;

    Sophus::SE3f camera_transformation_world_;

    std::shared_ptr<CameraModel> calibration_;

    int id_ = 0;

    float median_deformation_magnitud_;
};


#endif //NRSLAM_FRAME_H
