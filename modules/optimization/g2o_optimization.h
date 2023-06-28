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

#ifndef NRSLAM_G2O_OPTIMIZATION_H
#define NRSLAM_G2O_OPTIMIZATION_H

#include "map/frame.h"
#include "map/map.h"
#include "map/regularization_graph.h"

void CameraPoseOptimization(Frame& frame, const Sophus::SE3f& previous_camera_transform_world);

absl::flat_hash_set<ID> CameraPoseAndDeformationOptimization(Frame& current_frame,
                                                     std::shared_ptr<Map> map,
                                                     const Sophus::SE3f& previous_camera_transform_world,
                                                     const float scale);

absl::StatusOr<Eigen::Vector3f> DeformableTriangulation(TemporalBuffer& temporal_buffer,
                                                        int candidate_id,
                                                        std::shared_ptr<CameraModel> calibration,
                                                        const float scale);

void LocalDeformableBundleAdjustment(std::shared_ptr<Map> map,
                                     const float scale);


#endif //NRSLAM_G2O_OPTIMIZATION_H
