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

#include "map_visualizer.h"

#include <thread>

#include "absl/log/log.h"
#include "absl/log/check.h"

using namespace std;

MapVisualizer::MapVisualizer(Options& options, shared_ptr<Map> map) :
options_(options), map_(map), last_frame_id_drawn_(-1), last_frame_id_saved_(-1) {

}

void MapVisualizer::InitializePangolin() {
    pangolin::CreateWindowAndBind("DefSLAM",2*1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // Initial left view.
    Eigen::Matrix4f initial_left_view = options_.initial_left_view_;
    pangolin::OpenGlMatrix left_view;
    GLdouble* m1 = left_view.m;
#define M1(row,col)  m1[(col)*4+(row)]
    M1(0,0) = initial_left_view(0,0); M1(0,1) = initial_left_view(0,1); M1(0,2) = initial_left_view(0,2); M1(0,3) = initial_left_view(0,3);
    M1(1,0) = initial_left_view(1,0); M1(1,1) = initial_left_view(1,1); M1(1,2) = initial_left_view(1,2); M1(1,3) = initial_left_view(1,3);
    M1(2,0) = initial_left_view(2,0); M1(2,1) = initial_left_view(2,1); M1(2,2) = initial_left_view(2,2); M1(2,3) = initial_left_view(2,3);
    M1(3,0) = initial_left_view(3,0); M1(3,1) = initial_left_view(3,1); M1(3,2) = initial_left_view(3,2); M1(3,3) = initial_left_view(3,3);
#undef M

    // Initial right view.
    Eigen::Matrix4f initial_right_view = options_.initial_right_view_;
    pangolin::OpenGlMatrix right_view;
    GLdouble* m2 = right_view.m;
#define M2(row,col)  m2[(col)*4+(row)]
    M2(0,0) = initial_right_view(0,0); M2(0,1) = initial_right_view(0,1); M2(0,2) = initial_right_view(0,2); M2(0,3) = initial_right_view(0,3);
    M2(1,0) = initial_right_view(1,0); M2(1,1) = initial_right_view(1,1); M2(1,2) = initial_right_view(1,2); M2(1,3) = initial_right_view(1,3);
    M2(2,0) = initial_right_view(2,0); M2(2,1) = initial_right_view(2,1); M2(2,2) = initial_right_view(2,2); M2(2,3) = initial_right_view(2,3);
    M2(3,0) = initial_right_view(3,0); M2(3,1) = initial_right_view(3,1); M2(3,2) = initial_right_view(3,2); M2(3,3) = initial_right_view(3,3);
#undef M

    // Define Camera Render Object (for view / scene browsing).
    left_renderer_ = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
            left_view);

    right_renderer_ = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
            right_view);

    // Add named OpenGL viewport to window and provide 3D Handler
    left_display_ = pangolin::Display("[Left view] Map")
            .SetAspect(1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(left_renderer_));

    right_display = pangolin::Display("[Right view] Map")
            .SetAspect(1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(right_renderer_));

    main_display_ = pangolin::Display("multi")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(left_display_)
            .AddDisplay(right_display);

    // Set function to save rendered images.
    main_display_.extern_draw_function = [&](pangolin::View&) {
        if (map_->IsEmpty()) {
            return;
        }

        RenderLeftDisplay();

        RenderRightDisplay();
    };

    // Side menu.
    const int UI_WIDTH = 180;
    pangolin::CreatePanel("ui")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    show_ground_truth_ = shared_ptr<pangolin::Var<bool>>(
            new pangolin::Var<bool>("ui.Show GroundTruth", false, true));

}

void MapVisualizer::Run() {
    InitializePangolin();

    while (!should_finish) {
        RenderVisualization();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void MapVisualizer::RenderVisualization() {
    if (map_->IsEmpty()) {
        return;
    }

    ResetVisualization();

    RenderLeftDisplay();

    RenderRightDisplay();

    SaveRenderToDisk();

    FinishVisualization();
}

void MapVisualizer::ResetVisualization() {
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void MapVisualizer::RenderLeftDisplay() {
    left_display_.Activate(left_renderer_);

    DrawLastFrame();
    DrawKeyFrames();
    DrawLatestTrajectory();
    DrawNonTrackedLandmarks();
}

void MapVisualizer::RenderRightDisplay() {
    right_display.Activate(right_renderer_);

    DrawLastFrame();
    DrawKeyFrames();
    DrawLatestTrajectory();
    DrawNonTrackedLandmarks();
}

void MapVisualizer::FinishVisualization() {
    pangolin::FinishFrame();
}

void MapVisualizer::DrawLastFrame() {
    Frame last_frame = map_->GetLastFrame();
    last_frame_id_drawn_ = last_frame.GetId();

    vector<Eigen::Vector3f> landmarks_with_3d, landmarks_recently_triangulated;
    vector<vector<Eigen::Vector3f>> landmarks_with_3d_flow;
    vector<absl::StatusOr<Eigen::Vector3f>> landmarks_ground_truth;
    for (int idx = 0; idx < last_frame.LandmarkPositions().size(); idx++){
        if (last_frame.LandmarkStatuses()[idx] == TRACKED_WITH_3D) {
            auto landmark_id = last_frame.IndexToMapPointId().at(idx);
            auto landmark = map_->GetMapPoint(landmark_id);
            if (!landmark) {
                continue;
            }

            landmarks_with_3d.push_back(last_frame.LandmarkPositions()[idx]);
            landmarks_ground_truth.push_back(last_frame.GroundTruth()[idx]);

            landmarks_with_3d_flow.push_back(landmark->GetLandmarkFlow(20));
        } else if (last_frame.LandmarkStatuses()[idx] == JUST_TRIANGULATED) {
            landmarks_recently_triangulated.push_back(last_frame.LandmarkPositions()[idx]);
        }
    }

    Draw3DFlow(landmarks_with_3d_flow, Eigen::Vector3f(1, 0, 0));
    Draw3DPoints(landmarks_with_3d, Eigen::Vector3f(1, 0, 0));

    Draw3DPoints(landmarks_recently_triangulated, Eigen::Vector3f(1, 0, 0));

    if(*show_ground_truth_) {
        DrawGroundTruth(landmarks_with_3d, landmarks_ground_truth, Eigen::Vector3f(1, 0.5, 0.3));
    }

    DrawCamera(last_frame.CameraTransformationWorld(), Eigen::Vector3f(0, 1, 0));
}

void MapVisualizer::Draw3DPoints(std::vector<Eigen::Vector3f> point_cloud, Eigen::Vector3f color) {
    glPointSize(3);
    glColor3f(color.x(), color.y(), color.z());

    glBegin(GL_POINTS);
    for (Eigen::Vector3f landmark : point_cloud){
        glVertex3f(landmark(0), landmark(1), landmark(2));
    }

    glEnd();
}

void MapVisualizer::Draw3DFlow(const std::vector<std::vector<Eigen::Vector3f>> &flows, const Eigen::Vector3f color) {
    for (auto flow : flows) {
        if (flow.size() < 2) {
            continue;
        }
        DrawConnectedPoints(flow, color);
    }
}

void MapVisualizer::DrawGroundTruth(std::vector<Eigen::Vector3f> &estimated_point_cloud,
                                    std::vector<absl::StatusOr<Eigen::Vector3f>> &ground_truth_point_cloud,
                                    Eigen::Vector3f color) {
    glPointSize(3);
    glLineWidth(2);
    glColor3f(color.x(), color.y(), color.z());


    for (int idx = 0; idx < ground_truth_point_cloud.size(); idx++){
        if (ground_truth_point_cloud[idx].ok()) {
            const Eigen::Vector3f ground_truth_point = *ground_truth_point_cloud[idx];
            const Eigen::Vector3f estimated_landmark = estimated_point_cloud[idx];

            glBegin(GL_LINES);
            glVertex3f(ground_truth_point.x(), ground_truth_point.y(), ground_truth_point.z());
            glVertex3f(estimated_landmark.x(), estimated_landmark.y(), estimated_landmark.z());
            glEnd();

            glBegin(GL_POINTS);
            glVertex3f(ground_truth_point(0), ground_truth_point(1), ground_truth_point(2));
            glEnd();
        }

    }
}

void MapVisualizer::DrawCamera(Sophus::SE3f camera_transformation_world, Eigen::Vector3f color) {
    const float &w = options_.camera_size_;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    glMultMatrixf(camera_transformation_world.inverse().matrix().data());

    glLineWidth(2);
    glBegin(GL_LINES);
    glColor3f(color.x(), color.y(), color.z());

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void MapVisualizer::SetFinish() {
    should_finish = true;
}

void MapVisualizer::DrawKeyFrames() {
    auto keyframes = map_->GetKeyFrames();

    const int n_skip = 2;
    int n_from_last_drawn = 0;
    for (const auto& [id, keyframe] : keyframes) {
        if (n_from_last_drawn >= n_skip) {
            DrawCamera(keyframe->CameraTransformationWorld(), Eigen::Vector3f(0, 0, 1));
            n_from_last_drawn = 0;
        } else {
            n_from_last_drawn++;
        }
    }
}

void MapVisualizer::DrawLatestTrajectory() {
    auto latest_frames = map_->GetTemporalBuffer()->GetLatestCameraPoses();

    vector<Eigen::Vector3f> trajectory;
    for (auto& pose : latest_frames) {
        trajectory.push_back(pose.inverse().translation());
    }

    if (trajectory.size() < 2) {
        return;
    }

    DrawConnectedPoints(trajectory, Eigen::Vector3f(0, 0, 1));
}

void MapVisualizer::DrawConnectedPoints(std::vector<Eigen::Vector3f> &trajectory,
                                        const Eigen::Vector3f color) {
    CHECK_GE(trajectory.size(), 2);
    for(int idx = 0; idx < trajectory.size() - 1; idx++){
        Eigen::Vector3f current_point = trajectory[idx];
        Eigen::Vector3f next_point = trajectory[idx + 1];

        glBegin(GL_LINES);
        glColor3f(color.x(), color.y(), color.z());
        glVertex3f(current_point.x(), current_point.y(), current_point.z());
        glVertex3f(next_point.x(), next_point.y(), next_point.z());
        glEnd();
    }
}

void MapVisualizer::DrawNonTrackedLandmarks() {
    auto mappoints = map_->GetMapPoints();

    Frame last_frame = map_->GetLastFrame();

    vector<Eigen::Vector3f> non_tracked_3d, non_tracked_active_3d;
    vector<vector<Eigen::Vector3f>> non_tracked_active_3d_flow;
    for (const auto& [mappoint_id, mappoint] : mappoints) {
        if (last_frame.MapPointIdToIndex().contains(mappoint_id)) {
            continue;
        }

        if (mappoint->IsActive()) {
            non_tracked_active_3d.push_back(mappoint->GetLastWorldPosition());
            non_tracked_active_3d_flow.push_back(mappoint->GetLandmarkFlow(20));
        } else {
            non_tracked_3d.push_back(mappoint->GetLastWorldPosition());
        }
    }

    Draw3DPoints(non_tracked_3d, Eigen::Vector3f(0, 0, 0));
    Draw3DPoints(non_tracked_active_3d, Eigen::Vector3f(0, 0, 0));
}

void MapVisualizer::SaveRenderToDisk() {
    if (options_.render_save_path.empty()) {
        return;
    }

    if(last_frame_id_saved_ < last_frame_id_drawn_) {
        const string render_path = options_.render_save_path +
                to_string(last_frame_id_drawn_);
        main_display_.SaveRenderNow(render_path);

        last_frame_id_saved_ = last_frame_id_drawn_;
    }
}
