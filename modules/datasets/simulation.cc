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

#include "simulation.h"

#include <sys/stat.h>

#include "absl/log/log.h"
#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

using namespace std;

Simulation::Simulation(const std::string &dataset_path) {
    string names_file_path = dataset_path + "/rgb.txt";
    string detph_names_file_path = dataset_path + "/depth.txt";

    struct stat buffer;
    if(stat(names_file_path.c_str(),&buffer) != 0){   // Not processed.
        LOG(INFO) << "generating names file ... ";

        GenerateNamesFile(dataset_path);
    }

    ifstream names_file_reader;
    names_file_reader.open(dataset_path + "/rgb.txt");

    if(!names_file_reader.is_open()){
        LOG(FATAL) << "could not open names file at: " << dataset_path + "/names.txt";
        return;
    }

    images_names_.clear();

    while(!names_file_reader.eof()){
        string image_name;
        getline(names_file_reader, image_name);
        images_names_.push_back(image_name);
    }

    names_file_reader.close();

    ifstream depth_names_reader;
    depth_names_reader.open(dataset_path + "/depth.txt");

    if(!depth_names_reader.is_open()){
        LOG(FATAL) << "could not open names file at: " << dataset_path + "/depth.txt";
        return;
    }

    depth_images_names_.clear();

    while(!depth_names_reader.eof()){
        string image_name;
        getline(depth_names_reader, image_name);
        depth_images_names_.push_back(image_name);
    }

    depth_names_reader.close();

    ifstream poses_file_reader;
    poses_file_reader.open(dataset_path + "/trajectory.csv");

    if(!poses_file_reader.is_open()){
        LOG(FATAL) << "could not open trajectory file at: " << dataset_path + "/trajectory.csv";
        return;
    }

    string data;
    getline(poses_file_reader, data);

    ground_truth_poses_.clear();

    while(!poses_file_reader.eof()){
        getline(poses_file_reader, data);
        replace(data.begin(),data.end(),';',' ');

        stringstream ss;
        ss << data;

        //Format: tX;tY;tZ;rX;rY;rZ;rW;time(s)
        double timeStamp, vx, vy, vz, qw, qx, qy, qz;
        ss >> vx >> vy >> vz >> qx >> qy >> qz >> qw >> timeStamp;

        Sophus::SE3f pose(Eigen::Quaternionf(qw, qx, qy, qz),Eigen::Vector3f(vx, vy, vz));

        ground_truth_poses_.push_back(pose.inverse());
    }

    poses_file_reader.close();
}

absl::StatusOr<cv::Mat> Simulation::GetImage(const int idx) {
    if (idx >= images_names_.size()) {
        return absl::InternalError("Image index out boundaries.");
    }

    return cv::imread(images_names_[idx], cv::IMREAD_COLOR);
}

absl::StatusOr<cv::Mat> Simulation::GetDepthImage(const int idx) {
    if (idx >= depth_images_names_.size()) {
        return absl::InternalError("Image index out boundaries.");
    }

    cv::Mat depth_image = cv::imread(depth_images_names_[idx], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    cv::Mat channels[3];
    cv::split(depth_image,channels);

    depth_image = channels[2];

    float x = 1.f - far_clip_ / near_clip_;
    float y = far_clip_ / near_clip_;
    float z = x / far_clip_;
    float w = y / far_clip_;

    depth_image = 1.f / (z * (1 - depth_image) + w);

    return depth_image;
}

absl::StatusOr<Sophus::SE3f> Simulation::GetCameraPose(const int idx) {
    if (idx >= ground_truth_poses_.size()) {
        return absl::InternalError("Pose index out boundaries.");
    }

    return ground_truth_poses_[idx];
}

void Simulation::GenerateNamesFile(const std::string &images_path) {
    // Get number of images of the dataset.
    using namespace boost::filesystem;
    using namespace boost::lambda;

    const int n_files = std::count_if(
            directory_iterator(images_path + "/rgb"),
            directory_iterator(),
            static_cast<bool(*)(const path&)>(is_regular_file) );

    std::ofstream namesFile(images_path + "/rgb.txt");
    std::ofstream depthNamesFile(images_path + "/depth.txt");

    for(int i = 0; i < n_files; i++){
        namesFile << images_path << "/rgb/image_";
        depthNamesFile << images_path << "/depth/aov_image_";

        if(i < 10){
            namesFile << "000";
            depthNamesFile << "000";
        }
        else if(i >= 10 && i < 100){
            namesFile << "00";
            depthNamesFile << "00";
        }
        else if(i >= 100 && i < 1000){
            namesFile << "0";
            depthNamesFile << "0";
        }

        namesFile << i << ".png" << std::endl;
        depthNamesFile << i << ".exr" << std::endl;
    }

    namesFile.close();
    depthNamesFile.close();
}
