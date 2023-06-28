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

#ifndef NRSLAM_SYSTEM_H
#define NRSLAM_SYSTEM_H


#include <memory>
#include <thread>

#include "map/map.h"
#include "mapping/mapping.h"
#include "SLAM/settings.h"
#include "stereo/stereo_lucas_kanade.h"
#include "stereo/stereo_pattern_matching.h"
#include "tracking/tracking.h"
#include "utilities/frame_evaluator.h"
#include "utilities/time_profiler.h"
#include "visualization/image_visualizer.h"
#include "visualization/map_visualizer.h"

class System {
public:
    System() = delete;

    System(const std::string settings_file_path);

    ~System();

    // Tracks an image using the monocular pipeline
    void TrackImage(const cv::Mat& im);

    // Tracks the next image using some stereo information:
    //  - For stereo map initialization
    //  - Reconstruction evaluation
    // This is controlled by the settings passed to the system
    // TODO: implement stereo options
    void TrackImageWithStereo(const cv::Mat& im_left, const cv::Mat& im_right);

    // Tracks the next image using a precomputed depth image.
    void TrackImageWithDepth(const cv::Mat& im_left, const cv::Mat& im_depth);

private:
    // Applies preprocessing to the input image (CLAHE, etc).
    cv::Mat ImageProcessing(const cv::Mat& im, cv::Mat& im_gray);

    std::shared_ptr<Map> map_;

    std::unique_ptr<Tracking> tracker_;

    std::unique_ptr<Mapping> mapper_;

    std::unique_ptr<Settings> settings_;

    cv::Ptr<cv::CLAHE> clahe_;
    std::shared_ptr<Masker> masker_;

    // std::shared_ptr<StereoPatternMatching> stereo_matcher_;
    std::shared_ptr<StereoLucasKanade> stereo_matcher_;
    std::shared_ptr<StereoPatternMatching> stereo_pattern_matcher_;

    std::unique_ptr<MapVisualizer> map_visualizer_;
    std::unique_ptr<std::thread> map_visualizer_thread_;

    std::shared_ptr<ImageVisualizer> image_visualizer_;

    std::unique_ptr<FrameEvaluator> frame_evaluator_;

    std::unique_ptr<TimeProfiler> time_profiler_;
};

#endif //NRSLAM_SYSTEM_H
