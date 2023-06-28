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

#include "border_filter.h"

#include <opencv2/imgproc.hpp>

cv::Mat BorderFilter::generateMask(const cv::Mat &im){
    cv::Mat imGray = im;

    if (imGray.channels() == 3){
        cvtColor(imGray, imGray, cv::COLOR_BGR2GRAY);
    }
    else if (imGray.channels() == 4){
        cvtColor(imGray, imGray, cv::COLOR_BGR2GRAY);
    }
    cv::Rect roi(cb_, rb_, imGray.cols - ce_ - cb_, imGray.rows - re_ - rb_);
    cv::Mat mask = cv::Mat::zeros(imGray.size(), CV_8U);
    mask(roi) = cv::Scalar(255);
    mask.setTo(0, imGray == 0);
    cv::erode(mask, mask, getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));
    return mask;
}

std::string BorderFilter::getDescription(){
    return std::string("Border mask with parameters [" + std::to_string(rb_) + "," + std::to_string(re_) + "," +
                       std::to_string(cb_) + "," + std::to_string(ce_) + "]");
}