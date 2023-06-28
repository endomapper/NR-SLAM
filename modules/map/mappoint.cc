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

#include "mappoint.h"

using namespace std;

long unsigned int MapPoint::nextId_ = 0;

MapPoint::MapPoint(const Eigen::Vector3f &landmark_position, const int keypoint_id) {
    last_landmark_position_ = landmark_position;
    landmark_position_history_.push_back(landmark_position);
    keypoint_id_ = keypoint_id;
    is_active_ = false;

    id_ = nextId_++;
}

Eigen::Vector3f MapPoint::GetLastWorldPosition() {
    return last_landmark_position_;
}

void MapPoint::SetLastWorldPosition(Eigen::Vector3f &landmark_position) {
    last_landmark_position_ = landmark_position;

    landmark_position_history_.push_back(landmark_position);

    is_active_ = true;
}

void MapPoint::SetPhotometricInformation(LucasKanadeTracker::PhotometricInformation& photometric_information) {
    photometric_information_ = photometric_information;
}

LucasKanadeTracker::PhotometricInformation MapPoint::GetPhotometricInformation() {
    return photometric_information_;
}

int MapPoint::GetKeyPointId() {
    return keypoint_id_;
}

bool &MapPoint::IsActive() {
    return is_active_;
}

std::vector<Eigen::Vector3f> MapPoint::GetLandmarkFlow(const int n) {
    vector<Eigen::Vector3f>::iterator start_iterator;
    if (landmark_position_history_.size() < n) {
        start_iterator = landmark_position_history_.begin();
    } else {
        start_iterator = landmark_position_history_.begin() + (landmark_position_history_.size() - n);
    }

    auto end_iterator = landmark_position_history_.begin() + (landmark_position_history_.size());
    return vector<Eigen::Vector3f>(start_iterator, end_iterator);
}

long unsigned int MapPoint::GetId() {
    return id_;
}
