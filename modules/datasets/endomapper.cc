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

#include "endomapper.h"

#include <sys/stat.h>

#include <boost/filesystem.hpp>

#include "absl/log/log.h"

using namespace std;

Endomapper::Endomapper(const std::string &video_path) {
    string dataset_path = video_path.substr(0, video_path.find_last_of("/"));
    string output_images_directory = dataset_path + "/cam";

    // Checks if the video has been already split into single frames.
    struct stat buffer;
    if(stat(output_images_directory.c_str(), &buffer) != 0){   //Not processed
        LOG(INFO) << "splitting video into images... ";

        SplitVideoIntoFrames(dataset_path, video_path);
    } else{   //Already processed, just read the images names.
        LOG(INFO) << "loading already split dataset...";

        ifstream names_file;
        names_file.open(dataset_path + "/names.txt");

        if(!names_file.is_open()){
            LOG(FATAL) << "Could not open names file at: " << dataset_path + "/names.txt";
            return;
        }

        images_names_.clear();

        while(!names_file.eof()){
            string imName;
            getline(names_file, imName);
            images_names_.push_back(imName);
        }
    }
}

absl::StatusOr<cv::Mat> Endomapper::GetImage(const int idx) {
    if (idx >= images_names_.size()) {
        return absl::InternalError("Image index out boundaries.");
    }

    return cv::imread(images_names_[idx], cv::IMREAD_UNCHANGED);
}

bool Endomapper::SplitVideoIntoFrames(const std::string& dataset_path, const std::string& video_path) {
    // Opens the video.
    cv::VideoCapture video_capture(video_path);

    if(!video_capture.isOpened()){
        LOG(FATAL) << "Could not open video at: " << video_path;
        return false;
    }

    const int n_frames = video_capture.get(cv::CAP_PROP_FRAME_COUNT);

    // Opens file to save name files.
    ofstream names_file;
    names_file.open(dataset_path + "/names.txt");
    if(!names_file.is_open()){
        LOG(FATAL) << "Could not create names file at: " << dataset_path + "/names.txt";
        return false;
    }

    // Creates output directory.
    const string output_images_directory(dataset_path + "/cam");
    if(!boost::filesystem::create_directory(output_images_directory)){
        LOG(FATAL) << "Could not create output directory at: " << output_images_directory;
        return false;
    }

    int idx = 0;

    images_names_.clear();

    while(true){
        cv::Mat im;
        video_capture >> im;

        if(im.empty())
            break;

        const string image_name = output_images_directory + "/" + to_string(idx) + ".png";
        cv::imwrite(image_name,im);

        names_file << image_name << endl;
        images_names_.push_back(image_name);

        idx++;

        LOG(INFO) << "Processed image " << idx << " of " << n_frames;
    }

    return true;
}
