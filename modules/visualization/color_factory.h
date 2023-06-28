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
#ifndef NRSLAM_COLOR_FACTORY_H
#define NRSLAM_COLOR_FACTORY_H


#include <vector>

#include <opencv2/opencv.hpp>

class ColorFactory {
public:
    ColorFactory();

    std::vector<cv::Scalar> GetUniqueColors(const int n);
private:
    std::vector<cv::Scalar> unique_colors_;
};


#endif //NRSLAM_COLOR_FACTORY_H
