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

#ifndef NRSLAM_LUCAS_KANADE_TRACKER_H
#define NRSLAM_LUCAS_KANADE_TRACKER_H

#include "utilities/landmark_status.h"

#include <opencv2/opencv.hpp>

class LucasKanadeTracker {
public:
    struct PhotometricInformation {
        std::vector<float> mean_gray_per_level;
        std::vector<float> squared_mean_gray_per_level;
        std::vector<cv::Mat> gray_reference;
        std::vector<cv::Mat> gradient_reference;
    };

    /*
     * Default constructor
     */
    LucasKanadeTracker();

    /*
     * Constructor with parameters
     */
    LucasKanadeTracker(const cv::Size _winSize, const int _maxLevel, const int _maxIters,
                       const float _epsilon, const float _minEigThreshold);

    //-------------------------------------------------
    //   KLT with:
    //      -pre-computations
    //      -lighting invariance
    //-------------------------------------------------
    /*
     * Precomputes data for the reference image used for tracking next images
     */
    void SetReferenceImage(const cv::Mat &refIm, const std::vector<cv::KeyPoint> &refPts,
                           const cv::Mat& mask = cv::Mat());

    /*
     * Tracks a new image using precomputed data. SetReferenceImage must be called
     * at least once before calling this method
     */
    int Track(const cv::Mat &newIm, std::vector<cv::KeyPoint> &nextPts,
              std::vector<LandmarkStatus> &vMatched, const bool bInitialFlow,
              const float minSSIM, const cv::Mat& mask);

    PhotometricInformation GetPhotometricInformationOfPoint(const int idx);

    void InsertPhotometricInformation(cv::KeyPoint& keypoint, PhotometricInformation& photometric_information);

    void clear();

public:
    //-------------------------------
    //        KLT parameters
    //-------------------------------
    cv::Size winSize_;       //size of the integration window of the Lucas-Kanade algorithm
    int maxLevel_;           //max level of the image pyramids
    int maxIters_;           //max number of iterations of the optical flow algorithm
    float epsilon_;          //minimum optical flow imposed displacement. If lower, we stop computing
    float minEigThreshold_;  //min eigen threshold value for the Spatial Gradient matrix

    //-------------------------------
    //      Pre computed stuff
    //-------------------------------
    std::vector<cv::Mat> refPyr_;                //Reference pyramid
    std::vector<cv::Mat> mask_pyramid_;                //Mask pyramid
    std::vector<cv::KeyPoint> prevPts_;          //Original coordinates of the points to track
    std::vector<std::vector<float>> vMeanI_;     //Reference window mean intensities
    std::vector<std::vector<float>> vMeanI2_;    //Reference window mean squared intensities
    std::vector<std::vector<cv::Mat>> Iref_;     //Reference windows
    std::vector<std::vector<cv::Mat>> Idref_;    //Reference derivative windows
};


#endif //NRSLAM_LUCAS_KANADE_TRACKER_H
