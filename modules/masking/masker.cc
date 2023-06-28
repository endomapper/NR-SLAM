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
#include "masker.h"

#include "border_filter.h"
#include "bright_filter.h"
#include "predefined_filter.h"

#include "absl/log/log.h"

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

void Masker::loadFromTxt(std::string path){
    LOG(INFO) << "Loading filters: " << path;

    // Open filter file.
    std::ifstream filterFile(path);

    if (filterFile.is_open()){
        std::string line;
        while (getline(filterFile, line)){
            std::istringstream ss(line);

            //Get filter name
            std::string name;
            ss >> name;

            if (name == "BorderFilter"){
                std::string rb, re, cb, ce, th;
                ss >> rb >> re >> cb >> ce >> th;
                std::unique_ptr<Filter> f(new BorderFilter(stoi(rb), stoi(re), stoi(cb), stoi(ce), stoi(th)));
                addFilter(f);
            }
            else if (name == "BrightFilter"){
                std::string thLo;
                ss >> thLo;
                std::unique_ptr<Filter> f(new BrightFilter(stoi(thLo)));
                addFilter(f);
            }
            else if(name == "Predefined"){
                std::string path;
                ss >> path;
                std::unique_ptr<Filter> f(new PredefinedFilter(path));
                addFilter(f);
            }
        }

        filterFile.close();
    }
}

void Masker::addFilter(std::unique_ptr<Filter> &f){
    filters_.push_back(std::move(f));
}

void Masker::deleteFilter(size_t idx){
    filters_.erase(filters_.begin() + idx);
}

cv::Mat Masker::mask(const cv::Mat &im){
    // Generates an empty mask (all values set to 0).
    cv::Mat mask(im.rows, im.cols, CV_8U, cv::Scalar(255));

    // Apply each filter.
    for (auto &f : filters_){
        cv::bitwise_and(mask, f->generateMask(im), mask);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10));
    cv::erode(mask,mask,kernel);

    return mask.clone();
}

absl::flat_hash_map<std::string, cv::Mat> Masker::GetAllMasks(const cv::Mat &im){
    // Map with all the masks generated
    absl::flat_hash_map<std::string, cv::Mat> all_masks;

    // Generates an empty mask (all values set to 0).
    cv::Mat global_mask(im.rows, im.cols, CV_8U, cv::Scalar(255));

    // Apply each filter.
    for (auto &f : filters_){
        auto mask = f->generateMask(im);
        all_masks[f->GetFilterName()] = mask;

        cv::bitwise_and(global_mask, mask, global_mask);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10));
    cv::erode(global_mask,global_mask,kernel);

    all_masks["Global"] = global_mask;

    return all_masks;
}

std::string Masker::printFilters(){
    std::string msg("List of filters (" + std::to_string(filters_.size()) + "):\n");

    for (auto &f : filters_){
        msg += "\t-" + f->getDescription() + "\n";
    }

    return msg;
}