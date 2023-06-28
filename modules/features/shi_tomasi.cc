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

#include "shi_tomasi.h"

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace std::chrono;

ShiTomasi::ShiTomasi(Options& options) {
    non_max_suppresion_window_ = options.non_max_suprresion_window_size;
}

ShiTomasi::~ShiTomasi() {
    Xgrad.release();
    Ygrad.release();
    scores.release();
}

void ShiTomasi::Extract(const cv::Mat &im, std::vector<cv::KeyPoint> &keypoints) {
    if(im.cols != scores.cols || im.rows != scores.rows) {
        ResizeBuffers(im.size());
    }

    ComputeScores(im);

    if(keypoints.empty()){
        GetKeyPoints(keypoints);
    }
    else{
        vector<cv::KeyPoint> keypoints_already_extracted = keypoints;
        keypoints.clear();

        GetKeyPoints(keypoints, keypoints_already_extracted);
    }
}

void ShiTomasi::ResizeBuffers(cv::Size new_size) {
    Xgrad = cv::Mat::zeros(new_size, CV_16S);
    Ygrad = cv::Mat::zeros(new_size, CV_16S);
    scores = cv::Mat::zeros(new_size, CV_32F);
    r1.resize(new_size.width);
    r2.resize(new_size.width);
    r3.resize(new_size.width);

    nrows = new_size.height;
    ncols = new_size.width;
}

void ShiTomasi::ComputeScores(const cv::Mat& im){
    // Compute image gradients and, at the same time, the
    // Shi-Tomasi scores for each pixel. On this way, we are only
    // iterating over the image once.
    FastSobelXYandScore(im);
}

int ShiTomasi::GetKeyPoints(std::vector<cv::KeyPoint> &keypoints) {
    int points = 0;
    for(int r = 0; r < nrows; r++){
        for(int c = 0; c < ncols; c++){
            if(IsLocalMaximum(r,c)){
                cv::KeyPoint kp(c, r, 1);
                kp.class_id = next_feature_id_++;
                keypoints.push_back(kp);
                points++;
            }
        }
    }
    return points;
}

int ShiTomasi::GetKeyPoints(std::vector<cv::KeyPoint> &keypoints,
                            std::vector<cv::KeyPoint> &keypoints_already_extracted) {
    //First update scores to reflect already extracted points
    for(size_t i = 0; i < keypoints_already_extracted.size(); i++){
        scores.at<float>(round(keypoints_already_extracted[i].pt.y),
                            round(keypoints_already_extracted[i].pt.x)) = -1.f;
    }

    return this->GetKeyPoints(keypoints);
}

int ShiTomasi::GetPoints(std::vector<cv::Point2f> &points) {
    for(int r = 0; r < nrows; r++){
        for(int c = 0; c < ncols; c++){
            if(IsLocalMaximum(r,c)){
                points.push_back(cv::Point2f(c,r));
            }
        }
    }
    return points.size();
}

int ShiTomasi::GetPoints(std::vector<cv::Point2f> &points,
                         std::vector<cv::KeyPoint> &keypoints_already_extracted) {
    //First update scores to reflect already extracted points
    for(size_t i = 0; i < keypoints_already_extracted.size(); i++){
        scores.at<float>(round(keypoints_already_extracted[i].pt.y),
                            round(keypoints_already_extracted[i].pt.x)) = -1.f;
    }

    return this->GetPoints(points);
}

bool ShiTomasi::IsLocalMaximum(int r, int c) {
    const int NnoPrev = non_max_suppresion_window_;
    const int NPrev = 15;
    int minRow = max(0,r-NPrev);
    int minCol = max(0,c-NPrev);
    int maxRow = min(nrows-1,r+NPrev);
    int maxCol = min(ncols-1,c+NPrev);

    int minRow_inner = max(0,r-NnoPrev);
    int minCol_inner = max(0,c-NnoPrev);
    int maxRow_inner = min(nrows-1,r+NnoPrev);
    int maxCol_inner = min(ncols-1,c+NnoPrev);

    const float currValue = scores.at<float>(r,c);

    if(currValue == -1.f){
        return false;
    }

    if(currValue < 80){
        return false;
    }

    for(int i = minRow; i <= maxRow; i++){
        for(int j = minCol; j <= maxCol; j++){
            if(scores.at<float>(i,j) == -1.f)
                return false;

            if(i >= minRow_inner && i <= maxRow_inner &&
               j >= minCol_inner && j <= maxCol_inner){
                if(scores.at<float>(i,j) > currValue)
                    return false;
            }
        }
    }

    return true;
}


void ShiTomasi::FastSobelXYandScore(const cv::Mat& im){
    int rows_l = im.rows;
    int cols_l = im.cols;

    //########################
    //Grad X: First ROW
    //########################
    //First row
    pXgrad[1] = Xgrad.ptr<short>(0);
    pYgrad[1] = Ygrad.ptr<short>(0);

    pIm[1] = im.ptr<const uchar>(0);
    pIm[2] = im.ptr<const uchar>(1);

    //[First row]: first column

    //Compute short term data
    c1 = pIm[1][0]  + pIm[1][0] + pIm[2][0] + pIm[1][0];
    c2 = pIm[1][1]  + pIm[1][1] + pIm[2][1] + pIm[1][1];
    c3 = pIm[1][2]  + pIm[1][2] + pIm[2][2] + pIm[1][2];

    pXgrad[1][1] = c3 - c1;

    //[First row]: rest of columns
    for(int j = 2; j < rows_l - 1; j++){
        c1 = c2;
        c2 = c3;
        c3 = pIm[1][j+1] + pIm[1][j+1] + pIm[2][j+1] + pIm[2][j+1];

        pXgrad[1][j] = c3 - c1;
    }

    //########################
    //Second image ROW -> first of Y, also X
    //########################
    //Update pointers
    pXgrad[2] = Xgrad.ptr<short>(1);
    pYgrad[2] = Ygrad.ptr<short>(1);

    pIm[0] = pIm[1];
    pIm[1] = pIm[2];
    pIm[2] = im.ptr<uchar>(2);

    //Y GRAD -> FIRST COL
    r1[0] = pIm[0][0] + pIm[0][0] + pIm[0][1] + pIm[0][1];
    r2[0] = pIm[1][0] + pIm[1][0] + pIm[1][1] + pIm[1][1];
    r3[0] = pIm[2][0] + pIm[2][0] + pIm[2][1] + pIm[2][1];

    pYgrad[2][0] = r3[0] - r1[0];

    //SECOND COL: BOTH
    //X GRAD
    c1 = pIm[0][0]  + pIm[1][0] + pIm[1][0] + pIm[2][0];
    c2 = pIm[0][1]  + pIm[1][1] + pIm[1][1] + pIm[2][1];
    c3 = pIm[0][2]  + pIm[1][2] + pIm[1][2] + pIm[2][2];

    pXgrad[2][1] = c3 - c1;

    //Y GRAD
    r1[1] = pIm[0][0] + pIm[0][1] + pIm[0][1] + pIm[2][2];
    r2[1] = pIm[1][0] + pIm[1][1] + pIm[1][1] + pIm[1][2];
    r3[1] = pIm[2][0] + pIm[2][1] + pIm[2][1] + pIm[2][2];

    pYgrad[2][1] = r3[1] - r1[1];

    //X & Y GRAD -> INNER COLS
    for(int j = 2; j < cols_l - 1; j++){
        //X GRAD
        c1 = c2;
        c2 = c3;
        c3 = pIm[0][j + 1] + pIm[1][j + 1] + pIm[1][j + 1] + pIm[2][j + 1];

        pXgrad[2][j] = c3 - c1;

        //Y GRAD
        r1[j] = pIm[0][j - 1] + pIm[0][j] + pIm[0][j] + pIm[0][j + 1];
        r2[j] = pIm[1][j - 1] + pIm[1][j] + pIm[1][j] + pIm[1][j + 1];
        r3[j] = pIm[2][j - 1] + pIm[2][j] + pIm[2][j] + pIm[2][j + 1];

        pYgrad[2][j] = r3[j] - r1[j];
    }

    //Y GRAD LAST COLUMN
    r1[cols_l - 1] = pIm[0][cols_l - 1] + pIm[0][cols_l - 1] + pIm[0][cols_l - 2] + pIm[0][cols_l - 2];
    r2[cols_l - 1] = pIm[1][cols_l - 1] + pIm[1][cols_l - 1] + pIm[1][cols_l - 2] + pIm[1][cols_l - 2];
    r3[cols_l - 1] = pIm[2][cols_l - 1] + pIm[2][cols_l - 1] + pIm[2][cols_l - 2] + pIm[2][cols_l - 2];

    pYgrad[2][cols_l - 1] = r3[cols_l - 1] - r1[cols_l - 1];

    //########################
    //X & Y GRAD: INNER ROWS
    //From this point, we can start computing Shi-Tomasi scores
    //########################
    for(int i = 2; i < rows_l - 1; i++){
        //Update pointers
        pXgrad[0] = pXgrad[1];
        pXgrad[1] = pXgrad[2];
        pXgrad[2] = Xgrad.ptr<short>(i);

        pYgrad[0] = pYgrad[1];
        pYgrad[1] = pYgrad[2];
        pYgrad[2] = Ygrad.ptr<short>(i);

        pIm[0] = pIm[1];
        pIm[1] = pIm[2];
        pIm[2] = im.ptr<uchar>(i);

        r1 = r2;
        r2 = r3;

        //Y GRAD -> FIRST COL
        r3[0] = pIm[2][0] + pIm[2][0] + pIm[2][1] + pIm[2][1];

        pYgrad[2][0] = r3[0] - r1[0];

        //SECOND COL: BOTH
        //X GRAD
        c1 = pIm[0][0]  + pIm[1][0] + pIm[1][0] + pIm[2][0];
        c2 = pIm[0][1]  + pIm[1][1] + pIm[1][1] + pIm[2][1];
        c3 = pIm[0][2]  + pIm[1][2] + pIm[1][2] + pIm[2][2];

        pXgrad[2][1] = c3 - c1;

        //Y GRAD
        r3[1] = pIm[2][0] + pIm[2][1] + pIm[2][1] + pIm[2][2];

        pYgrad[2][1] = r3[1] - r1[1];

        pScore = scores.ptr<float>(i - 2);
        //X & Y GRAD -> INNER COLS
        for(int j = 2; j < cols_l - 1; j++){
            //X GRAD
            c1 = c2;
            c2 = c3;
            c3 = pIm[0][j + 1] + pIm[1][j + 1] + pIm[1][j + 1] + pIm[2][j + 1];

            pXgrad[2][j] = c3 - c1;

            //Y GRAD
            r3[j] = pIm[2][j - 1] + pIm[2][j] + pIm[2][j] + pIm[2][j + 1];

            pYgrad[2][j] = r3[j] - r1[j];

            //HERE WE CAN COMPUTE SHI-TOMASI SCORE
            DetectCorner(j - 1);
        }

        //Y GRAD LAST COLUMN
        r3[cols_l - 1] = pIm[2][cols_l - 1] + pIm[2][cols_l - 1] + pIm[2][cols_l - 2] + pIm[2][cols_l - 2];

        pYgrad[2][cols_l - 1] = r3[cols_l - 1] - r1[cols_l - 1];

        DetectCorner(cols_l - 2);
    }

    //########################
    //X GRAD: LAST ROW
    //########################
    pXgrad[0] = Xgrad.ptr<short>(rows_l - 3);
    pXgrad[1] = Xgrad.ptr<short>(rows_l - 2);
    pXgrad[2] = Xgrad.ptr<short>(rows_l - 1);

    pYgrad[0] = Ygrad.ptr<short>(rows_l - 3);
    pYgrad[1] = Ygrad.ptr<short>(rows_l - 2);
    pYgrad[2] = Ygrad.ptr<short>(rows_l - 1);

    //Last row -> first column
    c1 = pIm[1][0]  + pIm[1][0] + pIm[2][0] + pIm[1][0];
    c2 = pIm[1][1]  + pIm[1][1] + pIm[2][1] + pIm[1][1];
    c3 = pIm[1][2]  + pIm[1][2] + pIm[2][2] + pIm[1][2];

    pXgrad[2][1] = c3 - c1;

    for(int j = 1; j < rows_l - 1; j++){
        c1 = c2;
        c2 = c3;
        c3 = pIm[0][j + 1] + pIm[1][j + 1] + pIm[1][j + 1] + pIm[2][j + 1];

        pXgrad[2][j] = c3 - c1;

        DetectCorner(j);
    }
}

void ShiTomasi::DetectCorner(int col){
    if(col == 1){   //There are not any precomputed information
        //-------------------------------
        //Compute short term data
        //-------------------------------
        _G11_c1 = pXgrad[0][col] * pXgrad[0][col] + pXgrad[1][col] * pXgrad[1][col] + pXgrad[2][col] * pXgrad[2][col];
        _G12_c1 = pXgrad[0][col] * pYgrad[0][col] + pXgrad[1][col] * pYgrad[1][col] + pXgrad[2][col] * pYgrad[2][col];
        _G22_c1 = pYgrad[0][col] * pYgrad[0][col] + pYgrad[1][col] * pYgrad[1][col] + pYgrad[2][col] * pYgrad[2][col];

        _G11_c2 = pXgrad[0][col + 1] * pXgrad[0][col + 1] + pXgrad[1][col + 1] * pXgrad[1][col + 1] + pXgrad[2][col + 1] * pXgrad[2][col + 1];
        _G12_c2 = pXgrad[0][col + 1] * pYgrad[0][col + 1] + pXgrad[1][col + 1] * pYgrad[1][col + 1] + pXgrad[2][col + 1] * pYgrad[2][col + 1];
        _G22_c2 = pYgrad[0][col + 1] * pYgrad[0][col + 1] + pYgrad[1][col + 1] * pYgrad[1][col + 1] + pYgrad[2][col + 1] * pYgrad[2][col + 1];

        //-------------------------------
        //Compute spatial tensor
        //-------------------------------
        tensor[0] = (_G11_c1 + _G11_c2 + pXgrad[0][col - 1] * pXgrad[0][col - 1] + pXgrad[1][col - 1] * pXgrad[1][col - 1] + pXgrad[2][col - 1] * pXgrad[2][col - 1])  * inv_size;
        tensor[1] = (_G12_c1 + _G12_c2 + pXgrad[0][col - 1] * pYgrad[0][col - 1] + pXgrad[1][col - 1] * pYgrad[1][col - 1] + pXgrad[2][col - 1] * pYgrad[2][col - 1])  * inv_size;
        tensor[2] = (_G22_c1 + _G22_c2 + pYgrad[0][col - 1] * pYgrad[0][col - 1] + pYgrad[1][col - 1] * pYgrad[1][col - 1] + pYgrad[2][col - 1] * pYgrad[2][col - 1])  * inv_size;
    }
    else{   //We have information from the last 2 columns
        //-------------------------------
        //Compute partial tensor
        //-------------------------------
        tensor[0] = _G11_c1 + _G11_c2;
        tensor[1] = _G12_c1 + _G12_c2;
        tensor[2] = _G22_c1 + _G22_c2;

        //-------------------------------
        //Update short term data
        //-------------------------------
        _G11_c1 = _G11_c2;
        _G12_c1 = _G12_c2;
        _G22_c1 = _G22_c2;

        _G11_c2 = pXgrad[0][col + 1] * pXgrad[0][col + 1] + pXgrad[1][col + 1] * pXgrad[1][col + 1] + pXgrad[2][col + 1] * pXgrad[2][col + 1];
        _G12_c2 = pXgrad[0][col + 1] * pYgrad[0][col + 1] + pXgrad[1][col + 1] * pYgrad[1][col + 1] + pXgrad[2][col + 1] * pYgrad[2][col + 1];
        _G22_c2 = pYgrad[0][col + 1] * pYgrad[0][col + 1] + pYgrad[1][col + 1] * pYgrad[1][col + 1] + pYgrad[2][col + 1] * pYgrad[2][col + 1];

        //-------------------------------
        //Compute spatial tensor
        //-------------------------------
        tensor[0] = (tensor[0] + _G11_c2)  * inv_size;
        tensor[1] = (tensor[1] + _G12_c2)  * inv_size;
        tensor[2] = (tensor[2] + _G22_c2)  * inv_size;
    }

    //Compute min eigen value and save it
    ComputeMinEigenValue();
    pScore[col] = eigen_value;

    //Track the maximum score found
    max_score = max(eigen_value,max_score);
}

void ShiTomasi::ComputeMinEigenValue(){
    float tr = tensor[0] + tensor[2];
    float det = tensor[0] * tensor[2] - tensor[1] * tensor[1];

    float root = tr * tr - 4 * det;

    eigen_value = (tr - sqrt(root)) * 0.5;
}

void ShiTomasi::SetnonMaxSuppresionWindow(int windowSize) {
    non_max_suppresion_window_ = windowSize;
}
