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

#ifndef NRSLAM_SETTINGS_H
#define NRSLAM_SETTINGS_H

#include "calibration/camera_model.h"

#include "masking/masker.h"

#include <memory>

#include "sophus/se3.hpp"

class Settings {
public:
    /*
     * Default constructor: sets everything to default values
     */
    Settings();

    /*
     * Constructor reading parameters from file
     */
    Settings(const std::string& configFile);

    /*
     * Ostream operator overloading to dump settings to the terminal
     */
    friend std::ostream& operator<<(std::ostream& output, const Settings& D);

    //Getter methods
    std::shared_ptr<CameraModel> getCalibration();
    float getBf();

    std::shared_ptr<Masker> getMasker();

    float getRadPerPixel();

    Sophus::SE3f GetLeftMapVisualizationView();
    Sophus::SE3f GetRightMapVisualizationView();

    float GetCameraSize();

    bool GetAutoplay();

    std::string GetMapVisualizerPath();

    std::string GetImageVisualizerPath();

    std::string GetEvaluationPath();

private:
    //Camera parameters
    std::shared_ptr<CameraModel> calibration_;      //Geometric calibration with projection and unprojection functions
    float bf_;                                      //baseline times fx

    std::shared_ptr<Masker> masker_;

    float radPerPixel_;

    Sophus::SE3f left_map_view_, right_map_view_;

    float camera_size_;

    bool autoplay_;

    std::string map_visualizer_save_path_;

    std::string image_visualizer_save_path_;

    std::string evaluation_save_path_;

    template<typename T>
    T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return T();
            }

        }
        else{
            found = true;
            return (T) node;
        }
    }
};


#endif //NRSLAM_SETTINGS_H
