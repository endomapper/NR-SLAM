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

#include "monocular_map_initializer.h"

#include "utilities/dbscan.h"

#include "absl/log/log.h"

#include <Eigen/Core>

using namespace std;

MonocularMapInitializer::MonocularMapInitializer(Options& options,
                                                 std::shared_ptr<Feature> feature_extractor,
                                                 std::shared_ptr<CameraModel> calibration,
                                                 std::shared_ptr<ImageVisualizer> image_visualizer) :
    options_(options), feature_extractor_(feature_extractor), image_visualizer_(image_visualizer){
    klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                      options_.klt_max_level, options_.klt_max_iters,
                                      options_.klt_epsilon, options_.klt_min_eig_th);

    EssentialMatrixInitialization::Options essential_matrix_initializer_options;
    essential_matrix_initializer_options.max_features = options_.rigid_initializer_max_features;
    essential_matrix_initializer_options.min_sample_set_size = options_.rigid_initializer_min_sample_set_size;
    essential_matrix_initializer_options.min_parallax = options_.rigid_initializer_min_parallax;
    essential_matrix_initializer_options.radians_per_pixel = options_.rigid_initializer_radians_per_pixel;
    essential_matrix_initializer_options.epipolar_threshold = options_.rigid_initializer_epipolar_threshold;
    rigid_initializer_ = make_unique<EssentialMatrixInitialization>(essential_matrix_initializer_options, calibration, image_visualizer);

    calibration_ = calibration;

    internal_status_ = NO_DATA;
}

absl::StatusOr<MonocularMapInitializer::InitializationResults>
MonocularMapInitializer::ProcessNewImage(const cv::Mat& im, const cv::Mat& im_clahe,
                                              const cv::Mat& mask) {
    // Track features and update the feature tracks.
    DataAssociation(im, im_clahe, mask);

    if (internal_status_ == RECENTLY_RESET) {
        return absl::InternalError("Just reset");
    }

    // Perform optical flow clustering.
    auto feature_labels = FeatureTracksClustering();

    // Try to perform a rigid initialization.
    auto initialization_results_status = RigidInitialization();

    if (!initialization_results_status.ok()) {
        LOG(INFO) << initialization_results_status.status().message();
        return absl::InternalError("Rigid Initialization failed");
    }

    auto [camera_transform_world, landmarks_positions] = *initialization_results_status;

    // Perform a deformable Bundle Adjustment to refine the results.
    return InitializationRefinement(current_keypoints_, landmarks_positions, feature_labels, camera_transform_world);

}

void MonocularMapInitializer::ResetInitialization(const cv::Mat& im, const cv::Mat& im_clahe,
                                                  const cv::Mat& mask) {
    // Clear previous data.
    current_keypoints_.clear();

    feature_tracks_.max_feature_track_lenght = 0;
    feature_tracks_.feature_id_to_feature_track.clear();

    ExtractFeatures(im, mask, current_keypoints_);

    // Initialize KLT.
    klt_tracker_.SetReferenceImage(im, current_keypoints_);

    current_keypoint_statuses_.resize(current_keypoints_.size());
    fill(current_keypoint_statuses_.begin(), current_keypoint_statuses_.end(),
         TRACKED);

    images_from_last_reference_ = 0;

    // Set reference data in the rigid initializer.
    rigid_initializer_->ChangeReference(current_keypoints_);

    internal_status_ = RECENTLY_RESET;
}

void MonocularMapInitializer::DataAssociation(const cv::Mat& im, const cv::Mat& im_clahe,
                                              const cv::Mat& mask) {
    if (internal_status_ == NO_DATA) {
        ResetInitialization(im, im_clahe, mask);
    } else {
        // Track features.
        n_tracks_in_image_ = klt_tracker_.Track(im, current_keypoints_, current_keypoint_statuses_,
                           true, options_.klt_min_SSIM, mask);

        LOG(INFO) << "Number of matches: " << n_tracks_in_image_;

        if (n_tracks_in_image_ < 100) {
            ResetInitialization(im, im_clahe, mask);
        } else {
            images_from_last_reference_++;
            internal_status_ = OK;

            // Update KLT reference image if needed.
            if (images_from_last_reference_ > 30) {
                ResetInitialization(im, im_clahe, mask);

                images_from_last_reference_ = 0;
            }
        }
    }

    // Add features to the track history.
    AddFeatureTracks(current_keypoints_, current_keypoint_statuses_);
}

void MonocularMapInitializer::ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                               std::vector<cv::KeyPoint>& keypoints) {
    // Extract features.
    feature_extractor_->Extract(im, keypoints);

    // Mask out points.
    vector<cv::KeyPoint> masked_keypoints;
    for(size_t i = 0; i < keypoints.size(); i++){

        if(!mask.at<uchar>(keypoints[i].pt)){
            continue;
        }
        else{
            masked_keypoints.push_back(keypoints[i]);
        }
    }

    keypoints = masked_keypoints;
}

void MonocularMapInitializer::AddFeatureTracks(const std::vector<cv::KeyPoint> &keypoints,
                                               const std::vector<LandmarkStatus>& keypoint_statuses) {
    for (int idx = 0; idx < keypoints.size(); idx++) {
        if (keypoint_statuses[idx] == TRACKED) {
            const cv::KeyPoint keypoint = keypoints[idx];
            feature_tracks_.feature_id_to_feature_track[keypoint.class_id]
                .track_.push_back(keypoint);
        }
    }

    feature_tracks_.max_feature_track_lenght++;
}

void MonocularMapInitializer::UpdateTrackingReference(const cv::Mat& im) {
    vector<cv::KeyPoint> tracked_keypoints;
    for (int idx = 0; idx < current_keypoints_.size(); idx++) {
        if (current_keypoint_statuses_[idx] == TRACKED) {
            tracked_keypoints.push_back(current_keypoints_[idx]);
        }
    }

    current_keypoints_ = tracked_keypoints;

    current_keypoint_statuses_.resize(current_keypoints_.size());
    fill(current_keypoint_statuses_.begin(), current_keypoint_statuses_.end(),
         TRACKED);

    klt_tracker_.SetReferenceImage(im, current_keypoints_);
}

std::vector<int> MonocularMapInitializer::FeatureTracksClustering() {
    // Get only feature tracks with maximum length.
    const int max_track_length = feature_tracks_.max_feature_track_lenght;
    std::vector<Eigen::VectorXf> plain_feature_tracks;
    std::vector<std::vector<cv::Point2f>> feature_tracks;
    absl::flat_hash_map<int, int> idx_to_feature_id;
    for (const auto& [id, feature_track] : feature_tracks_.feature_id_to_feature_track) {
        if (feature_track.track_.size() == max_track_length) {
            idx_to_feature_id[feature_tracks.size()] = id;

            Eigen::VectorXf plain_track((max_track_length - 1) * 2);
            std::vector<cv::Point2f> track(max_track_length);

            track[0] = feature_track.track_[0].pt;

            for (int idx = 1; idx < feature_track.track_.size(); idx++) {
                cv::Point2f flow = feature_track.track_[idx].pt - feature_track.track_[idx - 1].pt;
                plain_track((idx - 1) * 2) = flow.x;
                plain_track((idx - 1) * 2 + 1) = flow.y;

                track[idx] = feature_track.track_[idx].pt;
            }

            plain_feature_tracks.push_back(plain_track);
            feature_tracks.push_back(track);
        }
    }

    vector<int> point_labels = DbscanND(plain_feature_tracks);

    // Draw clustered tracks.
    image_visualizer_->DrawClusteredOpticalFlow(feature_tracks, point_labels);

    return point_labels;
}

MonocularMapInitializer::RigidInitializationResults MonocularMapInitializer::RigidInitialization() {
    Sophus::SE3f camera_transform_world;
    std::vector<absl::StatusOr<Eigen::Vector3f>> landmarks_position;
    auto status = rigid_initializer_->Initialize(current_keypoints_, current_keypoint_statuses_,
                                                n_tracks_in_image_, camera_transform_world,
                                                landmarks_position);

    if (!status.ok()) {
        return absl::InternalError(status.message());
    } else {
        return make_tuple(camera_transform_world, landmarks_position);
    }
}

MonocularMapInitializer::InitializationResults
MonocularMapInitializer::InitializationRefinement(std::vector<cv::KeyPoint>& current_keypoints,
                              std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
                              std::vector<int>& feature_labels,
                              Sophus::SE3f& camera_transform_world) {
    vector<vector<cv::KeyPoint>> feature_tracks;
    vector<vector<Eigen::Vector3f>> landmark_tracks;
    vector<int> track_labels;
    vector<Sophus::SE3f> camera_trajectory;

    const int track_leghth = feature_tracks_.max_feature_track_lenght;

    for (int idx = 0; idx < landmarks_position.size(); idx++) {
        if (!landmarks_position[idx].ok()) {
            continue;
        }

        int feature_id = current_keypoints[idx].class_id;

        if (feature_tracks_.feature_id_to_feature_track[feature_id].track_.size() !=
            feature_tracks_.max_feature_track_lenght) {
            continue;
        }

        feature_tracks.push_back(feature_tracks_.feature_id_to_feature_track[feature_id].track_);
        track_labels.push_back(feature_labels[idx]);
        vector<Eigen::Vector3f> landmark_track(track_leghth, *landmarks_position[idx]);
        landmark_tracks.push_back(landmark_track);
    }

    // Interpolate camera trajectory.
    camera_trajectory.resize(track_leghth);
    Eigen::Quaternionf origin_rotation = Eigen::Quaternionf::Identity();
    for (int idx = 0; idx < track_leghth; idx++) {
        const float weight = idx / (track_leghth - 1);
        camera_trajectory[idx].translation() = camera_transform_world.translation() * weight;
        camera_trajectory[idx].setQuaternion(
                origin_rotation.slerp(weight, camera_transform_world.unit_quaternion()));
    }

    // Build reference and current frames from the estimated geometry.
    auto results = BuildInitializationResults(feature_tracks, landmark_tracks, track_labels, camera_trajectory);
    results.camera_transform_world = camera_transform_world;
    return results;
}

MonocularMapInitializer::InitializationResults
MonocularMapInitializer::BuildInitializationResults(std::vector<std::vector<cv::KeyPoint>> &feature_tracks,
                                            std::vector<std::vector<Eigen::Vector3f>> &landmark_tracks,
                                            std::vector<int> &track_labels,
                                            std::vector<Sophus::SE3f> &camera_trajectory) {
    MonocularMapInitializer::InitializationResults initialization_results;
    initialization_results.camera_transform_world = camera_trajectory.back();

    vector<float> initial_depths;

    for (int idx = 0; idx < feature_tracks.size(); idx++) {
        cv::KeyPoint reference_keypoint(feature_tracks[idx].front());
        cv::KeyPoint current_keypoint(feature_tracks[idx].back());

        Eigen::Vector3f reference_landmark_position = landmark_tracks[idx].front();
        Eigen::Vector3f current_landmark_position = landmark_tracks[idx].back();

        initial_depths.push_back(reference_landmark_position.z());

        initialization_results.reference_keypoints.push_back(reference_keypoint);
        initialization_results.current_keypoints.push_back(current_keypoint);
        initialization_results.reference_landmark_positions.push_back(reference_landmark_position);
        initialization_results.current_landmark_positions.push_back(current_landmark_position);
    }

    return initialization_results;
}
