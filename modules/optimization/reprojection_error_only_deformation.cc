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

#include "reprojection_error_only_deformation.h"

ReprojectionErrorOnlyDeformation::ReprojectionErrorOnlyDeformation() {};

bool ReprojectionErrorOnlyDeformation::read(std::istream& is) {
    return true;
}

bool ReprojectionErrorOnlyDeformation::write(std::ostream& os) const {
    return true;
}

void ReprojectionErrorOnlyDeformation::computeError()  {
    const LandmarkVertex* landmark_vertex = static_cast<const LandmarkVertex*>(_vertices[0]);
    Eigen::Vector2d obs(_measurement);  // Observed point in the image.
    Eigen::Vector3d landmark_poisition = landmark_vertex->estimate();

    _error = obs - calibration_->Project(landmark_poisition);
}
