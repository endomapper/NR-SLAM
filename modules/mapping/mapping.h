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

#ifndef NRSLAM_MAPPING_H
#define NRSLAM_MAPPING_H

#include "map/map.h"
#include "utilities/time_profiler.h"

class Mapping {
public:
    struct Options {
        float rad_per_pixel;
    };

    Mapping() = delete;

    Mapping(std::shared_ptr<Map> map, std::shared_ptr<CameraModel> calibration,
            const Options options, TimeProfiler* time_profiler);

    void DoMapping();

private:
    void KeyFrameMapping();

    void UpdateTrackingFrameFromKeyFrame(std::shared_ptr<KeyFrame> keyframe);

    void FrameMapping();

    void LandmarkTriangulation();

    absl::StatusOr<Eigen::Vector3f> DeformableLandmarkTriangulation(const int candidate_id);

    std::shared_ptr<Map> map_;

    std::shared_ptr<CameraModel> calibration_;

    TimeProfiler* time_profiler_;

    Options options_;
};


#endif //NRSLAM_MAPPING_H
