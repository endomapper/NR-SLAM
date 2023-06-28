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

#ifndef NRSLAM_MAP_H
#define NRSLAM_MAP_H

#include "map/frame.h"
#include "map/keyframe.h"
#include "map/mappoint.h"
#include "map/regularization_graph.h"
#include "map/temporal_buffer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

class RegularizationGraph;

class Map {
public:
    struct Options {
        int max_temporal_buffer_size = 20;
    };

    Map() = delete;

    Map(Options& options);

    // Inserts a new KeyFrame into the map.
    void InsertKeyFrame(std::shared_ptr<KeyFrame> keyframe);


    // Inserts a new MapPoint into the map.
    void InsertMapPoint(std::shared_ptr<MapPoint> mappoint);

    std::shared_ptr<MapPoint> CreateAndInsertMapPoint(const Eigen::Vector3f& position, const int keypoint_id);

    // Removes a MapPoint from the map.
    void RemoveMapPoint(ID id);


    // Gets all KeyFrames of the map.
    absl::btree_map<ID,std::shared_ptr<KeyFrame>> GetKeyFrames();


    // Gets all MapPoints of the map.
    absl::flat_hash_map<ID, std::shared_ptr<MapPoint>>& GetMapPoints();

    std::shared_ptr<MapPoint> GetMapPoint(ID id);

    std::shared_ptr<KeyFrame> GetKeyFrame(ID id);

    std::shared_ptr<KeyFrame> GetNextUnmappedKeyFrame();

    Frame GetLastFrame();

    std::shared_ptr<Frame> GetMutableLastFrame();

    void SetLastFrame(std::shared_ptr<Frame> frame);

    bool IsEmpty();

    void InitializeRegularizationGraph(const float sigma);

    std::shared_ptr<RegularizationGraph> GetRegularizationGraph();

    std::shared_ptr<TemporalBuffer> GetTemporalBuffer();

    void SetAllMappointsToNonActive();

    void SetMapScale(const float scale);

    float GetMapScale();

private:

    // Mappings of the KeyFrame/MapPoint ids and the KeyFrame/MapPoint itself.
    absl::flat_hash_map<ID, std::shared_ptr<MapPoint>> mappoints_;
    absl::btree_map<ID, std::shared_ptr<KeyFrame>> keyframes_;

    // List of the latest frames processed by the tracking.
    std::deque<ID> unmapped_keyframes_;

    // Temporal buffer for landmark triangulation
    std::shared_ptr<TemporalBuffer> temporal_buffer_;

    // Last frame for visualization
    std::shared_ptr<Frame> last_frame_;

    Frame frame_to_render_;

    // Regularization graph.
    std::shared_ptr<RegularizationGraph> regularization_graph_;

    Options options_;

    absl::Mutex last_frame_mutex_;
    absl::Mutex keyframes_mutex_;

    float map_scale_;
};


#endif //NRSLAM_MAP_H
