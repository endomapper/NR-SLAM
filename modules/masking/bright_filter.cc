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

#include "bright_filter.h"

#include <opencv2/imgproc.hpp>

cv::Mat BrightFilter::generateMask(const cv::Mat &im) {
    cv::Mat imGray = im, mask;
    if(imGray.channels()==3){
        cvtColor(imGray,imGray,cv::COLOR_BGR2GRAY);
    }
    else if(imGray.channels()==4){
        cvtColor(imGray,imGray,cv::COLOR_BGR2GRAY);
    }

    cv::threshold(imGray,mask,th_,255,cv::THRESH_BINARY_INV);

    cv::erode(mask, mask, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    cv::GaussianBlur(mask, mask, cv::Size(11, 11), 5, 5, cv::BORDER_REFLECT_101);

    return mask;
}

std::string BrightFilter::getDescription() {
    return std::string("Bright mask with th_ = " + std::to_string(th_));
}