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

#include "image_visualizer.h"

#include "absl/log/log.h"

using namespace std;

ImageVisualizer::ImageVisualizer(Options& options) : options_(options), current_image_number_(0) {}

void ImageVisualizer::SetCurrentImage(const cv::Mat& im_original, const cv::Mat& im_processed) {
    cv::cvtColor(im_processed, current_processed_image_, cv::COLOR_GRAY2BGR);

    current_color_image_ = im_original.clone();

    current_image_number_++;
}

void ImageVisualizer::DrawCurrentFrame(Frame &frame, const bool use_original_image) {
    cv::Mat image_to_display = (use_original_image) ? current_color_image_.clone() : current_processed_image_.clone();

    vector<cv::KeyPoint> keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});

    for (int idx = 0; idx < keypoints.size(); idx++) {
        DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 255,0));
    }

    cv::imshow("Current Frame", image_to_display);

    if (!options_.image_save_path.empty()) {
        string image_file_path = options_.image_save_path + "current_frame/"
                + to_string(frame.GetId()) + ".png";
        cv::imwrite(image_file_path, image_to_display);
    }
}

void ImageVisualizer::DrawFrame(Frame &frame, std::string name) {
    cv::Mat image_to_display = current_processed_image_.clone();

    vector<cv::KeyPoint> keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});

    for (int idx = 0; idx < keypoints.size(); idx++) {
        DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 255,0));
    }

    cv::imshow(name, image_to_display);
}

void ImageVisualizer::DrawRegularizationGraph(Frame &frame, RegularizationGraph &regularization_graph,
                                              const bool use_original_image) {
    cv::Mat image_to_display = (use_original_image) ? current_color_image_.clone() : current_processed_image_.clone();

    const int regularizers_per_point = 10;

    auto keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    auto mappoints_ids = frame.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    for (int idx = 0; idx < keypoints.size(); idx++) {
        DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 255, 0));

        const ID mappoint_id = mappoints_ids[idx];
        auto regularization_edges =
                regularization_graph.GetEdges(mappoint_id);

        int n_conections_drawn = 0;
        for(const auto& [mappoint_id_other, regularization_edge] : regularization_edges) {
            if (n_conections_drawn > regularizers_per_point ||
                regularization_edge->status == RegularizationGraph::BAD) {
                break;
            }

            // Check that the connected point is also being tracked.
            if (!frame.MapPointIdToIndex().contains(mappoint_id_other) ||
                    frame.LandmarkStatuses()[frame.MapPointIdToIndex().at(mappoint_id_other)] != TRACKED_WITH_3D) {
                continue;
            }

            const int idx_other = frame.MapPointIdToIndex().at(mappoint_id_other);

            auto color = HeatMapColor(regularization_graph.GetMinWeightAllowed(), 1.f, regularization_edge->weight);

            cv::line(image_to_display, keypoints[idx].pt, frame.Keypoints()[idx_other].pt,
                     color, 1, cv::LINE_AA);

            n_conections_drawn++;
        }


    }
    cout << endl;
    cv::imshow("Regularization graph", image_to_display);

    if (!options_.image_save_path.empty()) {
        string image_file_path = options_.image_save_path + "regularization_graph/"
                                 + to_string(frame.GetId()) + ".png";
        cv::imwrite(image_file_path, image_to_display);
    }
}

void ImageVisualizer::DrawOpticalFlow(TemporalBuffer &temporal_buffer) {
    cv::Mat image_to_display = current_processed_image_.clone();

    auto buffer = temporal_buffer.GetRawBuffer();

    TemporalBuffer::Snapshot last_snapshot = buffer.rbegin()->second;

    for (const auto& [track_id, keypoint] : last_snapshot.keypoint_tracks) {
        cv::Point2f last_pixel_coordinates = keypoint.pt;
        LandmarkStatus status = last_snapshot.keypoint_tracks_status[track_id];
        cv::Scalar color = (status == TRACKED) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);

        for (auto it = next(buffer.rbegin(), 1); it != buffer.rend(); it++) {
            TemporalBuffer::Snapshot snapshot = it->second;

            if (snapshot.keypoint_tracks.contains(track_id)) {
                // Draw optical flow segment
                cv::line(image_to_display, last_pixel_coordinates,
                         snapshot.keypoint_tracks[track_id].pt, color, 1);

                last_pixel_coordinates = snapshot.keypoint_tracks[track_id].pt;
            }
        }
    }

    cv::imshow("Optical flow", image_to_display);

    if (!options_.image_save_path.empty()) {
        string image_file_path = options_.image_save_path + "optical_flow/"
                                 + to_string(buffer.rbegin()->first) + ".png";
        cv::imwrite(image_file_path, image_to_display);
    }
}

void ImageVisualizer::DrawClusteredOpticalFlow(std::vector<std::vector<cv::Point2f>> &feature_tracks,
                                               std::vector<int> point_labels,
                                               const bool use_original_image) {
    cv::Mat image_to_display = (use_original_image) ? current_color_image_.clone() : current_processed_image_.clone();

    // Get how many cluster we have.
    absl::flat_hash_set<int> cluster_ids;
    for (auto id : point_labels) {
        cluster_ids.insert(id);
    }

    // Generate an unique color for each cluster.
    auto unique_colors = color_factory_.GetUniqueColors(50);
    absl::flat_hash_map<int, cv::Scalar> cluster_id_to_color;
    for (auto id : cluster_ids) {
        const cv::Scalar current_color = unique_colors[cluster_id_to_color.size()];
        cluster_id_to_color[id] = current_color;
    }

    for (int idx = 0; idx < feature_tracks.size(); idx++) {
        cv::Point2f last_pixel_coordinates = feature_tracks[idx].back();
        //cv::Scalar color = cluster_id_to_color[point_labels[idx]];
        cv::Scalar color = unique_colors[point_labels[idx] + 1];

        for (auto it = next(feature_tracks[idx].rbegin(), 1); it != feature_tracks[idx].rend(); it++) {
            cv::line(image_to_display, last_pixel_coordinates,
                     *it, color, 2);

            last_pixel_coordinates = *it;
        }
    }

    cv::imshow("Clustered optical flow", image_to_display);

    if (!options_.image_save_path.empty()) {
        string image_file_path = options_.image_save_path + "clustered_optical_flow/"
                                 + to_string(current_image_number_) + ".png";
        cv::imwrite(image_file_path, image_to_display);
    }
}

void ImageVisualizer::DrawFeatures(std::vector<cv::KeyPoint> &keypoints,
                                   const bool use_original_image) {
    cv::Mat image_to_display = (use_original_image) ? current_color_image_.clone() : current_processed_image_.clone();

    for (int idx = 0; idx < keypoints.size(); idx++) {
        DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 255,0));
    }

    cv::imshow("Essential Matrix inliers", image_to_display);

    if (!options_.image_save_path.empty()) {
        string image_file_path = options_.image_save_path + "current_features/"
                                 + to_string(current_image_number_) + ".png";
        cv::imwrite(image_file_path, image_to_display);
    }
}


void ImageVisualizer::DrawFeatures(std::vector<cv::KeyPoint> &keypoints,
                                   std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
                                   const bool use_original_image) {
    cv::Mat image_to_display = (use_original_image) ? current_color_image_.clone() : current_processed_image_.clone();

    for (int idx = 0; idx < keypoints.size(); idx++) {
        if (landmarks_position[idx].ok()) {
            DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 255,0));
        } else if (landmarks_position[idx].status().message() == "Internal triangulation error.") {
            DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 0, 255));
        } else if (landmarks_position[idx].status().message() == "Low parallax error.") {
            DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(255, 0, 0));
        } else if (landmarks_position[idx].status().message() == "High parallax error.") {
            DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(0, 255, 255));    // YELLOW
        } else if (landmarks_position[idx].status().message() == "Negative depth at first camera." ||
                   landmarks_position[idx].status().message() == "Negative depth at second camera.") {
            DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(255, 0, 255));  // MAGENTA
        } else {
            DrawFeature(image_to_display, keypoints[idx].pt, cv::Scalar(255, 255, 0));  // CYAN
        }
    }

    cv::imshow("Initialization results", image_to_display);
}

void ImageVisualizer::UpdateWindows() {
    if (options_.wait_for_user_button){
        cv::waitKey(0);
    } else {
        cv::waitKey(30);
    }

}

void ImageVisualizer::DrawFeature(cv::Mat &im, cv::Point2f uv, cv::Scalar color, int size) {
    cv::Point2f pt1, pt2;
    pt1.x=uv.x-size;
    pt1.y=uv.y-size;
    pt2.x=uv.x+size;
    pt2.y=uv.y+size;

    cv::rectangle(im,pt1,pt2,color);
    cv::circle(im,uv,2,color,-1);
}

cv::Scalar ImageVisualizer::HeatMapColor(const float min_value, const float max_value,
                                         const float value) {
    double v = value;
    double r = 1.0f, g = 1.0f, b = 1.0f;

    double dv;
    if (v < min_value)
        v = min_value;
    if (v > max_value)
        v = max_value;
    dv = max_value - min_value;

    if (v < (min_value + 0.25 * dv)) {
        r = 0;
        g = 4 * (v - min_value) / dv;
    }
    else if (v < (min_value + 0.5 * dv)) {
        r = 0;
        b = 1 + 4 * (min_value + 0.25 * dv - v) / dv;
    }
    else if (v < (min_value + 0.75 * dv)) {
        r = 4 * (v - min_value - 0.5 * dv) / dv;
        b = 0;
    }
    else {
        g = 1 + 4 * (min_value + 0.75 * dv - v) / dv;
        b = 0;
    }

    cv::Scalar color(b*255.f, g*255.f, r*255.f);

    return color;
}

cv::Scalar ImageVisualizer::OpenCVHeatMapColor(const float min_value, const float max_value, const float value,
                                               const int colormap) {
    cv::Mat dummy(1, 3, CV_8U), colored;
    dummy.at<uchar>(0, 0) = 0;
    dummy.at<uchar>(0, 1) = value * 255;
    dummy.at<uchar>(0, 2) = 255;

    cv::applyColorMap(dummy, colored, colormap);

    return cv::Scalar(colored.at<cv::Vec3b>(0, 1)[0],
                      colored.at<cv::Vec3b>(0, 1)[1],
                      colored.at<cv::Vec3b>(0, 1)[2]);
}

int ImageVisualizer::GetCurrentImageNumber() {
    return current_image_number_;
}
