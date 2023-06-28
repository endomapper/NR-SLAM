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

#ifndef NRSLAM_TEMPORAL_BUFFER_H
#define NRSLAM_TEMPORAL_BUFFER_H

#include "map/frame.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/btree_map.h"
#include "sophus/se3.hpp"

#include <opencv2/opencv.hpp>

typedef long unsigned int ID;

class TemporalBuffer {
public:
    struct Options {
        int max_buffer_size = 40;
    };

    TemporalBuffer() = delete;

    TemporalBuffer(Options& options);

    struct Snapshot {
        // Feature tracks.
        absl::flat_hash_map<int, cv::KeyPoint> keypoint_tracks;

        // Feature tracks status.
        absl::flat_hash_map<int, LandmarkStatus> keypoint_tracks_status;

        // Camera pose.
        Sophus::SE3f camera_transform_world;

        // MapPoint 3D tracks.
        absl::flat_hash_map<ID, Eigen::Vector3f> mapppoint_tracks_;

        // TODO: we should add index inside the frame to make things easier when inserting
        // in the frame after triangulation
        int frame_id;

        absl::flat_hash_map<int, int> keypoint_id_to_frame_idx;

        float deformation_magnitud;
    };

    void InsertSnapshotFromFrame(Frame& frame);

    absl::btree_map<ID, Snapshot>& GetRawBuffer();

    std::vector<int> GetTriangulationCandidatesIds();

    int TrackLenght(const int keypoint_id);

    std::vector<Sophus::SE3f> GetLatestCameraPoses();

    std::vector<int> GetClosestMapPointsToFeature(const int keypoint_id,
                                                  const int num_neighbors,
                                                  const int min_image_distance,
                                                  const int max_image_distance);

    std::vector<std::pair<float, int>> GetClosestMapPointsToLocation(const Eigen::Vector3f location,
                                                   const int keypoint_id,
                                                   const int num_neighbors);

    std::vector<std::pair<ID, cv::KeyPoint>>GetFeatureTrack(const int keypoint_id);

    absl::StatusOr<Sophus::SE3f> GetCameraTransformWorld(const int frame_id);

    absl::StatusOr<Eigen::Vector3f> GetLandmarkPosition(const int frame_id,
                                                        const int keypoint_id);

    absl::StatusOr<int> GetLandmarkIndexInFrame(const int frame_id,
                                                const int keypoint_id);

    bool CheckRigidity(const int first_frame_id, const int last_frame_id,
                       const float rigidity_th);

private:
    absl::btree_map<ID, Snapshot> buffer_;

    Options options_;

};


#endif //NRSLAM_TEMPORAL_BUFFER_H
