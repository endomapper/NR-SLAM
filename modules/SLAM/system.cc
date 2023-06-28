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

#include "system.h"

#include "absl/log/log.h"

using namespace std;

System::System(const string settings_file_path) {
    // Output welcome message
    LOG(INFO).NoPrefix() << "NR-SLAM Copyright (C) Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.";
    LOG(INFO).NoPrefix() << "This program comes with ABSOLUTELY NO WARRANTY;";
    LOG(INFO).NoPrefix() << "This is free software, and you are welcome to redistribute it";
    LOG(INFO).NoPrefix() << "under certain conditions. See LICENSE.txt.";

    settings_ = make_unique<Settings>(settings_file_path);
    LOG(INFO) << *settings_;

    // Initialize image processing stuff
    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
    masker_ = settings_->getMasker();

    // Create map
    Map::Options map_options;
    map_options.max_temporal_buffer_size = 20;
    map_ = make_shared<Map>(map_options);

    StereoLucasKanade::Options stereo_matcher_options;
    stereo_matcher_options.klt_window_size = 21;
    stereo_matcher_options.klt_max_level = 4;
    stereo_matcher_options.klt_max_iters = 10;
    stereo_matcher_options.klt_epsilon = 0.0001;
    stereo_matcher_options.klt_min_eig_th = 0.0001;
    stereo_matcher_options.klt_min_SSIM = 0.5;

    stereo_matcher_ = make_shared<StereoLucasKanade>(stereo_matcher_options, settings_->getCalibration(),
                                                     settings_->getBf());

    stereo_pattern_matcher_ = make_shared<StereoPatternMatching>(settings_->getCalibration(),
                                                                 settings_->getBf());

    // Initialize map visualizer and launch it in a different thread
    MapVisualizer::Options map_visualizer_options;
    map_visualizer_options.camera_size_ = settings_->GetCameraSize();
    map_visualizer_options.initial_left_view_ = settings_->GetLeftMapVisualizationView().matrix();
    map_visualizer_options.initial_right_view_ = settings_->GetRightMapVisualizationView().matrix();
    map_visualizer_options.render_save_path = settings_->GetMapVisualizerPath();

    map_visualizer_ = make_unique<MapVisualizer>(map_visualizer_options, map_);

    map_visualizer_thread_ = make_unique<thread>(&MapVisualizer::Run, map_visualizer_.get());

    //Initialize image visualizer.
    ImageVisualizer::Options image_visualizer_options;
    image_visualizer_options.wait_for_user_button = !settings_->GetAutoplay();
    image_visualizer_options.image_save_path = settings_->GetImageVisualizerPath();
    image_visualizer_ = make_shared<ImageVisualizer>(image_visualizer_options);

    // Initialize Tracking.
    Tracking::Options tracking_options;
    tracking_options.klt_window_size = 21;
    tracking_options.klt_max_level = 4;
    tracking_options.klt_max_iters = 10;
    tracking_options.klt_epsilon = 0.0001;
    tracking_options.klt_min_eig_th = 0.0001;
    tracking_options.klt_min_SSIM = 0.7;
    tracking_options.radians_per_pixel = settings_->getRadPerPixel();

    // Time profiler.
    time_profiler_ = make_unique<TimeProfiler>();

    tracker_ = make_unique<Tracking>(tracking_options, map_, settings_->getCalibration(),
                                     stereo_matcher_, image_visualizer_, time_profiler_.get());

    // Initialize Mapping.
    Mapping::Options mapping_options;
    mapping_options.rad_per_pixel = settings_->getRadPerPixel();
    mapper_ = make_unique<Mapping>(map_, settings_->getCalibration(), mapping_options, time_profiler_.get());

    // Initialize frame evaluator.
    FrameEvaluator::Options frame_evaluator_options;
    frame_evaluator_options.results_file_path = settings_->GetEvaluationPath();
    frame_evaluator_options.precomputed_depth_ = true;
    frame_evaluator_ = make_unique<FrameEvaluator>(frame_evaluator_options, stereo_pattern_matcher_,
                                                   map_visualizer_.get());
}

System::~System() {
    // Send signal to the visualizer to finish
    map_visualizer_->SetFinish();

    // Wait until is done
    map_visualizer_thread_->join();
}

void System::TrackImage(const cv::Mat &im) {
    // Preprocess image.
    cv::Mat im_gray;
    cv::Mat processed_image = ImageProcessing(im, im_gray);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im, processed_image);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray);

    // Perform tracking.
    tracker_->TrackImage(im_gray, masks, cv::Mat(), processed_image);

    // Perform mapping.
    mapper_->DoMapping();

    // Draw images.
    image_visualizer_->UpdateWindows();
}

void System::TrackImageWithStereo(const cv::Mat &im_left, const cv::Mat &im_right) {
    // Preprocess images.
    cv::Mat im_gray_left, im_gray_right;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);
    cv::Mat processed_image_right = ImageProcessing(im_right, im_gray_right);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    // Perform tracking.
    tracker_->TrackImage(im_gray_left, masks, im_gray_right, processed_image_left);

    // Perform mapping.
    mapper_->DoMapping();

    // Evaluate reconstruction.
    if (false && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_gray_right);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();
}

void System::TrackImageWithDepth(const cv::Mat &im_left, const cv::Mat &im_depth) {
    // Preprocess images.
    cv::Mat im_gray_left, im_gray_right;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    // Perform tracking.
    tracker_->TrackImage(im_gray_left, masks, im_gray_right, processed_image_left);

    // Perform mapping.
    mapper_->DoMapping();

    // Evaluate reconstruction.
    if (true && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_depth);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();
}

cv::Mat System::ImageProcessing(const cv::Mat &im, cv::Mat& im_gray) {
    cv::Mat processed_image;

    // Convert to grayscale.
    cv::cvtColor(im, processed_image, cv::COLOR_RGB2GRAY);

    im_gray = processed_image.clone();

    // Apply Clahe to the image.
    clahe_->apply(processed_image, processed_image);

    return processed_image;
}

