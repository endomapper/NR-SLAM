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

#include "hamlyn.h"

#include <fstream>
#include <sys/stat.h>

#include <boost/filesystem.hpp>

#include "absl/log/log.h"

using namespace std;

Hamlyn::Hamlyn(const std::string &video_path, const std::string& other_video_path) {
    string path = video_path.substr(0,video_path.find_last_of("/"));
    string images_dir = path + "/left";

    //Check if the video has been already split into single frames
    struct stat buffer;
    if(stat(images_dir.c_str(),&buffer) != 0){   //Not processed
        LOG(INFO) << "Splitting video into frames...";

        SplitVideoIntoFrames(path, video_path, other_video_path);
    }
    else{   //Already processed, just read images names
        LOG(INFO) << "Loading already split dataset";

        ifstream names_file_reader;
        names_file_reader.open(path + "/namesLeft.txt");

        if(!names_file_reader.is_open()){
            LOG(ERROR) << "Could not open names file at: " << path + "/names.txt";
            return;
        }

        left_images_names_.clear();

        while(!names_file_reader.eof()){
            string image_name;
            getline(names_file_reader, image_name);
            left_images_names_.push_back(image_name);
        }

        names_file_reader.close();

        ifstream right_names_file_reader;
        right_names_file_reader.open(path + "/namesRight.txt");

        if(!right_names_file_reader.is_open()){
            LOG(ERROR) << "Could not open names file at: " << path + "/names.txt";
            return;
        }

        right_images_names.clear();

        while(!right_names_file_reader.eof()){
            string image_name;
            getline(right_names_file_reader, image_name);
            right_images_names.push_back(image_name);
        }

        right_names_file_reader.close();

    }
}

absl::StatusOr<cv::Mat> Hamlyn::GetImage(const int idx) {
    if(idx >= left_images_names_.size()) {
        return absl::InternalError("Image index out boundaries.");
    }

    return cv::imread(left_images_names_[idx], cv::IMREAD_UNCHANGED);
}

absl::StatusOr<cv::Mat> Hamlyn::GetRightImage(const int idx) {
    if(idx >= right_images_names.size()) {
        return absl::InternalError("Image index out boundaries.");
    }

    return cv::imread(right_images_names[idx], cv::IMREAD_UNCHANGED);
}

bool Hamlyn::SplitVideoIntoFrames(const std::string &path, const std::string &video_path,
                                  const std::string& other_video_path) {
    const bool has_two_videos = !other_video_path.empty();

    // Open video.
    cv::VideoCapture video_capture(video_path);

    if (!video_capture.isOpened()){
        LOG(FATAL) << "Could not open video at: " << video_path;
        return false;
    }

    const int n_frames = video_capture.get(cv::CAP_PROP_FRAME_COUNT);

    // Open other video if it exists.
    cv::VideoCapture other_video_capture;
    if (has_two_videos) {
        other_video_capture = cv::VideoCapture(other_video_path);

        if (!other_video_capture.isOpened()) {
            LOG(FATAL) << "Could not open video at: " << other_video_path;
            return false;
        }
    }

    // Open file to save name files.
    ofstream left_names_writer, right_names_writer;
    left_names_writer.open(path + "/namesLeft.txt");
    right_names_writer.open(path + "/namesRight.txt");
    if(!left_names_writer.is_open()){
        LOG(FATAL) << "Could not create names file at: " << path + "/names.txt";
        return false;
    }

    // Create output directory.
    const string left_images_path(path + "/left");
    if(!boost::filesystem::create_directory(left_images_path)){
        LOG(FATAL) << "Could not create output directory at: " << left_images_path;
        return false;
    }

    const string right_images_path(path + "/right");
    if(!boost::filesystem::create_directory(right_images_path)){
        LOG(FATAL) << "Could not create output directory at: " << right_images_path;
        return false;
    }

    int idx = 0;

    left_images_names_.clear();
    right_images_names.clear();

    // Use this for Hamlyn 20-21.
    leftCal_ = (cv::Mat_<double>(3,3) << 755.312744, 0.0, 327.875,
            0.0, 420.477722, 165.484406,
            0.0, 0.0, 1.0);
    leftDistorsion_ = (cv::Mat_<double>(1,4) << -0.186853, 0.122769, -0.010146, -0.003869);

    rightCal_ = (cv::Mat_<double>(3,3) << 759.047791, 0.0, 391.990051,
            0.0, 415.329529, 151.748993,
            0.0, 0.0, 1);
    rightDistorsion_ = (cv::Mat_<double>(1,4) << -0.197641, 0.213583, -0.00037, -0.010498);

    R = (cv::Mat_<double>(3,3) << 0.999835, 0.001024, 0.018154,
            -0.001085, 0.999994, 0.003314,
            -0.018151, -0.003333, 0.99983);
    t = (cv::Mat_<double>(3,1) << -5.196155,
            -0.030411,
            0.212897);

    cv::Size leftImSize(720,288);

    cv::Size newSize = cv::Size(leftImSize.width,leftImSize.height*1.79);

    // Use this for Hamlyn 01.
    /* leftCal_ = (cv::Mat_<double>(3,3) << 381.914307, 0.0, 168.108963,
            0.0, 383.797882, 126.979446,
            0.0, 0.0, 1.0);
    leftDistorsion_ = (cv::Mat_<double>(1,4) << -0.333236, 0.925076, 0.003847, 0.000916);

    rightCal_ = (cv::Mat_<double>(3,3) << 381.670013, 0.0, 129.929291,
            0.0, 382.582397, 120.092186,
            0.0, 0.0, 1);
    rightDistorsion_ = (cv::Mat_<double>(1,4) << -0.329342, 0.699034, 0.004927, 0.008194);

    R = (cv::Mat_<double>(3,3) << 0.999906, 0.006813, -0.011930,
            -0.006722, 0.999948, 0.007680,
            0.011981, -0.007599, 0.999899);
    t = (cv::Mat_<double>(3,1) << 5.382236, 0.067659, -0.039156);

    cv::Size leftImSize(320,240);

    cv::Size newSize = cv::Size(2 * leftImSize.width, 2 * leftImSize.height);*/

    cv::stereoRectify(leftCal_, leftDistorsion_, rightCal_, rightDistorsion_, leftImSize, R, t, R_l, R_r, P_l, P_r, Q, cv::CALIB_ZERO_DISPARITY, -1, newSize, 0, 0);
    cv::initUndistortRectifyMap(leftCal_, leftDistorsion_, R_l, P_l.rowRange(0, 3).colRange(0, 3),
                                newSize, CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(rightCal_, rightDistorsion_, R_r, P_r.rowRange(0, 3).colRange(0, 3),
                                newSize, CV_32F, M1r, M2r);

    LOG(INFO) << "P_l: " << P_l;
    LOG(INFO) << "P_r: " << P_r;

    while(true){
        cv::Mat left_image, right_image;
        if(!has_two_videos) {
            cv::Mat image;
            video_capture >> image;

            if (image.empty())
                break;

            int rows = image.rows;
            int cols = image.cols;
            cv::Rect left_ROI(0, 0, cols / 2, rows);
            cv::Rect right_ROI(cols / 2, 0, cols / 2, rows);
            left_image = image(left_ROI);
            right_image = image(right_ROI);
        } else {
            video_capture >> left_image;
            other_video_capture >> right_image;

            if (left_image.empty() || right_image.empty())
                break;
        }

        // Rectify images.
        cv::Mat left_image_rectified, right_image_rectified;
        cv::remap(left_image, left_image_rectified, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(right_image, right_image_rectified, M1r, M2r, cv::INTER_LINEAR);

        // Save images.
        const string left_image_name = left_images_path + "/" + to_string(idx) + ".png";
        cv::imwrite(left_image_name, left_image_rectified);

        left_names_writer << left_image_name << endl;
        left_images_names_.push_back(left_image_name);

        const string right_image_name = right_images_path + "/" + to_string(idx) + ".png";
        cv::imwrite(right_image_name, right_image_rectified);

        right_names_writer << right_image_name << endl;
        right_images_names.push_back(right_image_name);

        idx++;
        LOG(INFO) << "Processed image " << idx << " of " << n_frames;
    }

    return true;
}
