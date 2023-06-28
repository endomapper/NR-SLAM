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

#ifndef NRSLAM_TRACKING_H
#define NRSLAM_TRACKING_H

#include "calibration/camera_model.h"
#include "features/feature.h"
#include "map/frame.h"
#include "map/map.h"
#include "matching/lucas_kanade_tracker.h"
#include "stereo/stereo_lucas_kanade.h"
#include "stereo/stereo_pattern_matching.h"
#include "tracking/monocular_map_initializer.h"
#include "utilities/time_profiler.h"
#include "visualization/image_visualizer.h"

#include "absl/container/flat_hash_set.h"

class Tracking {
public:
    struct Options {
        int klt_window_size = 21;
        int klt_max_level = 3;
        int klt_max_iters = 50;
        float klt_epsilon = 0.01;
        float klt_min_eig_th = 1e-4;
        float klt_min_SSIM = 0.7;

        int images_to_insert_keyframe = 5;

        float radians_per_pixel;
    };

    enum TrackingStatus {
        NOT_INITIALIZED,
        TRACKING,
        LOST
    };

    Tracking() = delete;

    Tracking(const Options options, std::shared_ptr<Map> map,
             std::shared_ptr<CameraModel> calibration,
             std::shared_ptr<StereoLucasKanade> stereo_matcher,
             std::shared_ptr<ImageVisualizer> image_visualizer,
             TimeProfiler* time_profiler);

    void TrackImage(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                    const cv::Mat& additional_im = cv::Mat(), const cv::Mat& im_clahe = cv::Mat());

    TrackingStatus GetTrackingStatus() const;

private:
    void ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                         std::vector<cv::KeyPoint>& keypoints);

    void MonocularMapInitialization(const cv::Mat& im_left,
                                 const cv::Mat& mask, const cv::Mat& im_clahe);

    void StereoMapInitialization(const cv::Mat& im_left, const cv::Mat& im_right,
                                 const cv::Mat& mask, const cv::Mat& im_clahe);

    absl::flat_hash_set<ID> TrackCameraAndDeformation(const cv::Mat& im, const cv::Mat& mask);

    void DataAssociation(const cv::Mat& im, const cv::Mat& mask);

    void CameraPoseEstimation();

    absl::flat_hash_set<ID> CameraPoseAndDeformationEstimation();

    void KeyFrameInsertion(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks);

    bool NeedNewKeyFrame();

    void CreateNewKeyFrame(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks);

    void ExtractFeaturesInFrame(const cv::Mat& im, const cv::Mat& mask, Frame& frame);

    void SetKLTReference(const cv::Mat& im, Frame& frame, const cv::Mat& mask);

    void PointReuse(const cv::Mat& im, const cv::Mat& mask,
                    absl::flat_hash_set<ID> lost_mappoint_ids);

    void UpdateTriangulatedPoints();

    Options options_;

    std::shared_ptr<Map> map_;

    std::shared_ptr<CameraModel> calibration_;

    std::shared_ptr<Feature> feature_extractor_;

    LucasKanadeTracker klt_tracker_;

    std::shared_ptr<Frame> current_frame_;

    //std::shared_ptr<StereoPatternMatching> stereo_matcher_;
    std::shared_ptr<StereoLucasKanade> stereo_matcher_;

    Sophus::SE3f motion_model_;

    std::shared_ptr<ImageVisualizer> image_visualizer_;

    int n_images_from_last_keyframe_ = 0;

    std::unique_ptr<MonocularMapInitializer> monocular_map_initializer_;

    TrackingStatus tracking_status_;

    Sophus::SE3f previous_camera_transform_world_;

    TimeProfiler* time_profiler_;
};


#endif //NRSLAM_TRACKING_H
