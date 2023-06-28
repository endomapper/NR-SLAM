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

#include "tracking.h"

#include "features/shi_tomasi.h"
#include "optimization/g2o_optimization.h"
#include "utilities/dbscan.h"
#include "utilities/geometry_toolbox.h"
#include "utilities/statistics_toolbox.h"

#include "absl/log/log.h"
#include "absl/log/check.h"


using namespace std;

Tracking::Tracking(const Tracking::Options options, std::shared_ptr<Map> map,
                   std::shared_ptr<CameraModel> calibration,
                   std::shared_ptr<StereoLucasKanade> stereo_matcher,
                   std::shared_ptr<ImageVisualizer> image_visualizer,
                   TimeProfiler* time_profiler) :
options_(options), map_(map), calibration_(calibration), stereo_matcher_(stereo_matcher),
image_visualizer_(image_visualizer), tracking_status_(NOT_INITIALIZED), time_profiler_(time_profiler)
{
    ShiTomasi::Options shi_tomasi_options;
    shi_tomasi_options.non_max_suprresion_window_size = 7;
    feature_extractor_ = make_shared<ShiTomasi>(shi_tomasi_options);

    klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                      options_.klt_max_level, options_.klt_max_iters,
                                      options_.klt_epsilon, options_.klt_min_eig_th);

    current_frame_ = make_shared<Frame>();

    current_frame_->SetCalibration(calibration);

    MonocularMapInitializer::Options monocular_map_initializer_options;
    monocular_map_initializer_options.klt_window_size = 21;
    monocular_map_initializer_options.klt_max_level = 4;
    monocular_map_initializer_options.klt_max_iters = 10;
    monocular_map_initializer_options.klt_epsilon = 0.0001;
    monocular_map_initializer_options.klt_min_eig_th = 0.0001;
    monocular_map_initializer_options.klt_min_SSIM = 0.5;

    monocular_map_initializer_options.rigid_initializer_max_features = 4000;
    monocular_map_initializer_options.rigid_initializer_min_sample_set_size = 8;
    monocular_map_initializer_options.rigid_initializer_min_parallax = 0.999;
    monocular_map_initializer_options.rigid_initializer_radians_per_pixel = options_.radians_per_pixel;
    monocular_map_initializer_options.rigid_initializer_epipolar_threshold = 0.005;

    monocular_map_initializer_ = make_unique<MonocularMapInitializer>(
            monocular_map_initializer_options, feature_extractor_, calibration_, image_visualizer_);
}

void Tracking::TrackImage(const cv::Mat &im, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                          const cv::Mat &additional_im, const cv::Mat& im_clahe) {
    map_->SetAllMappointsToNonActive();

    if (map_->IsEmpty()) {
        // If map is not initialized, perform map initialization.

        // For stereo experiment purposes.
        // StereoMapInitialization(im, additional_im, masks.at("Global"), im_clahe);

        // Depending on the type of sequence, the mask type used can be different.
        MonocularMapInitialization(im, masks.at("Global"), im_clahe);
        // MonocularMapInitialization(im, masks.at("PredefinedFilter"), im_clahe);
    } else {
        // Update points triangulated by the mapping in the last frame.
        UpdateTriangulatedPoints();

        // Otherwise perform normal tracking.
        // Depending on the type of sequence, the mask type used can be different.
        // absl::flat_hash_set<ID> lost_mappoint_ids = TrackCameraAndDeformation(im, masks.at("BorderFilter"));
        absl::flat_hash_set<ID> lost_mappoint_ids = TrackCameraAndDeformation(im, masks.at("Global"));

        // Point reuse.
        PointReuse(im, cv::Mat(), lost_mappoint_ids);

        if (current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size() < 10) {
            exit(0);
        }

        // KeyFrame insertion.
        KeyFrameInsertion(im, masks);

        // Insert frame to the temporal buffer.
        map_->SetLastFrame(current_frame_);

        // Draw current frame.
        image_visualizer_->DrawCurrentFrame(*current_frame_);
        image_visualizer_->DrawRegularizationGraph(*current_frame_, *(map_->GetRegularizationGraph()));
        image_visualizer_->DrawFeatures(current_frame_->Keypoints());
    }
}

Tracking::TrackingStatus Tracking::GetTrackingStatus() const {
    return tracking_status_;
}

void Tracking::ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                               std::vector<cv::KeyPoint>& keypoints) {
    // Extract features.
    feature_extractor_->Extract(im, keypoints);

    // Mask out points.
    vector<cv::KeyPoint> masked_keypoints;
    for(size_t i = 0; i < keypoints.size(); i++){
        if(!mask.at<uchar>(keypoints[i].pt)){
            continue;
        } else{
            masked_keypoints.push_back(keypoints[i]);
        }
    }

    keypoints = masked_keypoints;
}

void Tracking::MonocularMapInitialization(const cv::Mat& im_left,
                                       const cv::Mat& mask, const cv::Mat& im_clahe) {
    auto initialization_status = monocular_map_initializer_->ProcessNewImage(im_left, im_clahe, mask);

    if(!initialization_status.ok()) {
        LOG(INFO) << initialization_status.status().message();
        return;
    }

    auto initialization_results = *initialization_status;

    vector<float> depths;
    for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx];
        depths.push_back(current_landmark_position.z());
    }

    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    const float median_depth = depths[median_idx];
    const float scale = 3.f / median_depth;
    map_->SetMapScale(scale);

    float sigma = Sigma(depths);
    float sigma_scaled = sigma * scale;

    Frame reference_frame;
    for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
        cv::KeyPoint reference_keypoint = initialization_results.reference_keypoints[idx];
        cv::KeyPoint current_keypoint = initialization_results.current_keypoints[idx];

        Eigen::Vector3f reference_landmark_position = initialization_results.reference_landmark_positions[idx] * scale;
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx] * scale;

        ID mappoint_id = map_->CreateAndInsertMapPoint(reference_landmark_position,
                                                       reference_keypoint.class_id)->GetId();

        reference_frame.InsertObservation(reference_keypoint,
                                          reference_landmark_position,
                                          mappoint_id,
                                          TRACKED_WITH_3D);

        current_frame_->InsertObservation(current_keypoint,
                                          current_landmark_position,
                                          mappoint_id,
                                          TRACKED_WITH_3D);
    }
    reference_frame.SetCalibration(calibration_);

    reference_frame.MutableCameraTransformationWorld() = Sophus::SE3f();
    initialization_results.camera_transform_world.translation() = initialization_results.camera_transform_world.translation() * scale;
    current_frame_->MutableCameraTransformationWorld() = initialization_results.camera_transform_world;

    // Create Keyframes from the frames.
    auto first_keyframe = make_shared<KeyFrame>(reference_frame);
    auto current_keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(first_keyframe);
    map_->InsertKeyFrame(current_keyframe);

    map_->SetLastFrame(current_frame_);

    // Initialize regularization graph.
    map_->InitializeRegularizationGraph(sigma_scaled * 3);

    // Set reference image to the KLT tracker.
    klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());

    // Save MapPoint photometric information
    for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }

    tracking_status_ = TRACKING;
}

void Tracking::StereoMapInitialization(const cv::Mat& im_left, const cv::Mat& im_right,
                                       const cv::Mat& mask, const cv::Mat& im_clahe) {
    current_frame_->Clear();

    vector<cv::KeyPoint> keypoints;
    ExtractFeatures(im_clahe, mask, keypoints);

    std::vector<Eigen::Vector3f> filtered_landmarks;
    std::vector<cv::KeyPoint> filtered_keypoints;

    auto stereo_matcher = StereoPatternMatching(calibration_, 3886.37);
    for (int idx = 0; idx < keypoints.size(); idx++) {
        auto landmark = stereo_matcher.computeStereo3D(keypoints[idx], im_left, im_right);
        if (landmark.ok() &&
            (*landmark).z() > 35.5f && (*landmark).z() < 70.5f) {
            filtered_landmarks.push_back(*landmark);
            filtered_keypoints.push_back(keypoints[idx]);
        }
    }

    // Apply dbscan to remove further outliers
    vector<int> labels = Dbscan3D(filtered_landmarks);

    std::vector<float> depths;

    for (int idx = 0; idx < labels.size(); idx++) {
        if (labels[idx] == 0) {
            depths.push_back(filtered_landmarks[idx].z());
        }
    }

    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    float median_depth = depths[median_depth];
    const float scale = 1.f;

    current_frame_->MutableCameraTransformationWorld().translation() *= scale;

    for(int idx = 0; idx < labels.size(); idx++){
        if (labels[idx] == 0) {
            auto mappoint = map_->CreateAndInsertMapPoint(filtered_landmarks[idx] * scale,
                                                          filtered_keypoints[idx].class_id);
            current_frame_->InsertObservation(filtered_keypoints[idx],
                                             filtered_landmarks[idx],
                                             mappoint->GetId(),
                                             TRACKED_WITH_3D);
        }
    }

    map_->SetLastFrame(current_frame_);

    // Initialize regularization graph.
    map_->InitializeRegularizationGraph(10.5);
    map_->SetMapScale(1.f);

    // Set reference image to the KLT tracker.
    klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());

    // Save MapPoint photometric information
    for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }

    // Create Keyframe from the current frame.
    auto keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(keyframe);

    tracking_status_ = TRACKING;
}

absl::flat_hash_set<ID> Tracking::TrackCameraAndDeformation(const cv::Mat &im, const cv::Mat& mask) {
    // Perform data association.
    DataAssociation(im, mask);

    // Coarse camera pose estimation.
    CameraPoseEstimation();

    // Deformation + camera pose estimation.
    auto lost_ids = CameraPoseAndDeformationEstimation();
    return lost_ids;
}

void Tracking::DataAssociation(const cv::Mat &im, const cv::Mat &mask) {
    klt_tracker_.Track(im, current_frame_->Keypoints(), current_frame_->LandmarkStatuses(),
                       true, options_.klt_min_SSIM, mask);
}

void Tracking::CameraPoseEstimation() {
    // Apply motion model to get a first seed of the current camera pose.
    current_frame_->MutableCameraTransformationWorld() = motion_model_ *
            current_frame_->CameraTransformationWorld();

    previous_camera_transform_world_ = current_frame_->CameraTransformationWorld();

    // Do optimization.
    CameraPoseOptimization(*current_frame_, previous_camera_transform_world_);
}

absl::flat_hash_set<ID> Tracking::CameraPoseAndDeformationEstimation() {
    // Do optimization.
    auto lost_mappoint_ids = CameraPoseAndDeformationOptimization(*current_frame_,
                                         map_,previous_camera_transform_world_,
                                         map_->GetMapScale());

    // Update motion model.
    motion_model_ = current_frame_->CameraTransformationWorld() *
                    map_->GetLastFrame().CameraTransformationWorld().inverse();

    return lost_mappoint_ids;
}

void Tracking::KeyFrameInsertion(const cv::Mat& im,
                                 const absl::flat_hash_map<std::string, cv::Mat>& masks) {
    if (NeedNewKeyFrame()) {
        CreateNewKeyFrame(im, masks);
    }
}

bool Tracking::NeedNewKeyFrame() {
    if(n_images_from_last_keyframe_ >= options_.images_to_insert_keyframe){
        n_images_from_last_keyframe_ = 0;
        return true;
    }
    else{
        n_images_from_last_keyframe_++;
        return false;
    }
}

void Tracking::CreateNewKeyFrame(const cv::Mat& im,
                                 const absl::flat_hash_map<std::string, cv::Mat>& masks) {
    // Extract new features.
    ExtractFeaturesInFrame(im, masks.at("Global"), *current_frame_);

    // Create Keyframe from the current frame.
    auto keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(keyframe);

    // Update current frame.
    current_frame_->SetFromKeyFrame(keyframe);

    // Set new klt reference.
    // Depending on the type of sequence, the mask type used can be different.
    // SetKLTReference(im, *current_frame_, masks.at("BorderFilter"));
    SetKLTReference(im, *current_frame_, masks.at("Global"));
    // SetKLTReference(im, *current_frame_, cv::Mat());
    // SetKLTReference(im, *current_frame_, masks.at("PredefinedFilter"));
}

void Tracking::ExtractFeaturesInFrame(const cv::Mat& im, const cv::Mat& mask, Frame &frame) {
    vector<cv::KeyPoint> tracked_keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D, TRACKED});

    ExtractFeatures(im, mask, tracked_keypoints);

    for (int idx = 0; idx < tracked_keypoints.size(); idx++) {
        frame.InsertObservation(tracked_keypoints[idx], Eigen::Vector3f::Zero(), 0, TRACKED);
    }
}

void Tracking::SetKLTReference(const cv::Mat& im, Frame& frame, const cv::Mat& mask) {
    klt_tracker_.SetReferenceImage(im, frame.Keypoints(), mask);

    // Update MapPoints photometric information.
    for (const auto &[mappoint_id, idx] : frame.MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }
}

void Tracking::PointReuse(const cv::Mat& im, const cv::Mat& mask,
                          absl::flat_hash_set<ID> lost_mappoint_ids) {
    auto all_mappoints = map_->GetMapPoints();
    for (const auto &[mappoint_id, mappoint] : all_mappoints) {
        if (!current_frame_->LandmarkPosition(mappoint_id).ok()) {
            // Project mappoint into the camera and check if it lies inside the image.
            Eigen::Vector3f landmark_position_seed = mappoint->GetLastWorldPosition();
            Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position_seed;

            if (landmark_camera_position.z() < 0) {
                continue;
            }

            cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

            if (projected_landmark.x >= 0 && projected_landmark.x < im.cols &&
                projected_landmark.y >= 0 && projected_landmark.y < im.rows) {
                lost_mappoint_ids.insert(mappoint_id);
            }
        }
    }

    if (lost_mappoint_ids.empty()) {
        return;
    }

    // Project candidates into the image.
    Frame frame_with_only_candidates;

    LucasKanadeTracker klt(cv::Size(options_.klt_window_size, options_.klt_window_size),
                           1, options_.klt_max_iters,
                           options_.klt_epsilon, options_.klt_min_eig_th);

    int candidates_in_image = 0;
    vector<cv::KeyPoint> keypoint_seeds;
    for (const auto& mappoint_id : lost_mappoint_ids) {
        auto mappoint = map_->GetMapPoint(mappoint_id);
        Eigen::Vector3f landmark_position_seed = mappoint->GetLastWorldPosition();
        Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position_seed;
        cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

        if (isnan(projected_landmark.x) || isnan(projected_landmark.y)) {
            LOG(FATAL) << "NaN found!";
        }

        if (projected_landmark.x >= 0 && projected_landmark.x < im.cols &&
            projected_landmark.y >= 0 && projected_landmark.y < im.rows) {
            cv::KeyPoint keypoint(projected_landmark, 1);
            frame_with_only_candidates.InsertObservation(keypoint, landmark_position_seed, mappoint_id, TRACKED_WITH_3D);

            // Set photometric information in the KLT.
            LucasKanadeTracker::PhotometricInformation photometric_information =
                    map_->GetMapPoint(mappoint_id)->GetPhotometricInformation();
            klt.InsertPhotometricInformation(keypoint, photometric_information);

            keypoint_seeds.push_back(keypoint);

            candidates_in_image++;
        }
    }

    if (candidates_in_image == 0) {
        return;
    }

    // Track candidates with KLT
    klt.Track(im, frame_with_only_candidates.Keypoints(), frame_with_only_candidates.LandmarkStatuses(),
              true, 0.75, mask);

    // Insert tracked candidates into the current frame
    vector<cv::KeyPoint> tracked_candidate_keypoints =
            frame_with_only_candidates.GetKeypointsWithStatus({TRACKED_WITH_3D});
    vector<Eigen::Vector3f> tracked_candidate_landmarks =
            frame_with_only_candidates.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    vector<ID> tracked_candidate_mappoint_ids =
            frame_with_only_candidates.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    int reused_landmarks = 0;

    for (int idx = 0; idx < tracked_candidate_keypoints.size(); idx++) {
        cv::KeyPoint keypoint = tracked_candidate_keypoints[idx];
        Eigen::Vector3f landmark_position = tracked_candidate_landmarks[idx];
        ID mappoint_id = tracked_candidate_mappoint_ids[idx];

        keypoint.class_id = map_->GetMapPoint(mappoint_id)->GetKeyPointId();

        Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position;
        cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

        if (SquaredReprojectionError(projected_landmark, keypoint.pt) > 5.99) {
            continue;
        }

        if (current_frame_->MapPointIdToIndex().contains(mappoint_id)) {
            const int idx_in_frame = current_frame_->MapPointIdToIndex().at(mappoint_id);

            current_frame_->Keypoints()[idx_in_frame] = keypoint;
            current_frame_->LandmarkPositions()[idx_in_frame] = landmark_position;
            current_frame_->LandmarkStatuses()[idx_in_frame] = TRACKED_WITH_3D;

        } else {
            current_frame_->InsertObservation(keypoint, landmark_position, mappoint_id, TRACKED_WITH_3D);

            LucasKanadeTracker::PhotometricInformation photometric_information =
                    map_->GetMapPoint(mappoint_id)->GetPhotometricInformation();
            klt_tracker_.InsertPhotometricInformation(keypoint, photometric_information);
        }

        reused_landmarks++;
    }

    LOG(INFO) << "Reused landmarks: " << reused_landmarks;
}

void Tracking::UpdateTriangulatedPoints() {
    auto indices = current_frame_->GetIndexWithStatus({JUST_TRIANGULATED});

    for (auto index : indices) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(index);

        ID mappoint_id = current_frame_->IndexToMapPointId().at(index);
        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);

        // Update landmark status.
        current_frame_->LandmarkStatuses()[index] = TRACKED_WITH_3D;
    }
}
