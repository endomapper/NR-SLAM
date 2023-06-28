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

#include "lucas_kanade_tracker.h"

#include <float.h>

#include <numeric>
#include <boost/assert.hpp>

using namespace std;
using namespace cv;

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

LucasKanadeTracker::LucasKanadeTracker() : winSize_(Size(21, 21)), maxLevel_(3), maxIters_(30), epsilon_(0.01),
                                           minEigThreshold_(1e-4) {}

LucasKanadeTracker::LucasKanadeTracker(const cv::Size winSize, const int maxLevel, const int maxIters,
                                       const float epsilon, const float minEigThreshold) :
        winSize_(winSize), maxLevel_(maxLevel), maxIters_(maxIters), epsilon_(epsilon),
        minEigThreshold_(minEigThreshold) {
    vMeanI_ = vector<vector<float>>(maxLevel + 1);
    vMeanI2_ = vector<vector<float>>(maxLevel + 1);
    Iref_ = vector<vector<Mat>>(maxLevel + 1);
    Idref_ = vector<vector<Mat>>(maxLevel + 1);

}

void LucasKanadeTracker::SetReferenceImage(const Mat &refIm, const vector<KeyPoint> &refPts,
                                           const cv::Mat& mask) {
    //Compute reference pyramid
    cv::buildOpticalFlowPyramid(refIm, refPyr_, winSize_, maxLevel_);

    //Store points
    prevPts_ = refPts;

    //Compute reference windows (intensity and derivtives) and means of the windows
    Point2f halfWin((winSize_.width - 1) * 0.5f, (winSize_.height - 1) * 0.5f);

    const int borderGap = round(winSize_.width/2);

    for (int level = maxLevel_; level >= 0; level--) {
        vMeanI_[level].resize(refPts.size(),-1);
        vMeanI2_[level].resize(refPts.size(),-1);
        Iref_[level].resize(refPts.size(),cv::Mat());
        Idref_[level].resize(refPts.size(),cv::Mat());

        //Get images form the pyramid
        const Mat I = refPyr_[level * 2];
        const Mat derivI = refPyr_[level * 2 + 1];

        //Steps for matrix indexing
        int dstep = (int) (derivI.step / derivI.elemSize1());
        int stepI = (int) (I.step / I.elemSize1());

        //Buffer for fast memory access
        int cn = I.channels(), cn2 = cn * 2;   //cn should be 1 therefor cn2 should be 2
        AutoBuffer<short> _buf(winSize_.area() * (cn + cn2));
        int derivDepth = DataType<short>::depth;

        //Integration window buffers
        Mat IWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn), (*_buf));
        Mat derivIWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn2), (*_buf) + winSize_.area() * cn);

        const int scaleFactor = (1 << level);

        for (int i = 0; i < prevPts_.size(); i++) {
            //Compute image coordinates in the reference image at the current level
            Point2f point = prevPts_[i].pt / (float)(1 << level);

            Point2i ipoint;
            point -= halfWin;
            ipoint.x = cvFloor(point.x);
            ipoint.y = cvFloor(point.y);

            if (ipoint.x < -borderGap || ipoint.x >= derivI.cols - borderGap ||
                ipoint.y < -borderGap || ipoint.y >= derivI.rows - borderGap) {
                continue;
            }

            //Compute weighs for sub pixel computation
            float a = point.x - ipoint.x;
            float b = point.y - ipoint.y;
            const int W_BITS = 14, W_BITS1 = 14;
            const float FLT_SCALE = 1.f / (1 << 20);
            int iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
            int iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
            int iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            //Compute sumI, sumI2, meanI, meanI2, Iwin, IdWin
            float meanI = 0.f, meanI2 = 0.f;

            int x, y;
            bool valid = true;
            for (y = 0; y < winSize_.height; y++) {
                //Get pointers to the images
                const uchar *src = I.ptr() + (y + ipoint.y) * stepI + ipoint.x;
                const short *dsrc = derivI.ptr<short>() + (y + ipoint.y) * dstep + ipoint.x * 2;

                //Get pointers to the window buffers
                short *Iptr = IWinBuf.ptr<short>(y);
                short *dIptr = derivIWinBuf.ptr<short>(y);

                x = 0;
                for (; x < winSize_.width * cn; x++, dsrc += 2, dIptr += 2) {
                    int mx = (ipoint.x+x) * scaleFactor;
                    int my = (ipoint.y+y) * scaleFactor;

                    if(!mask.empty() && mask.at<uchar>(my,mx) == 0){
                        valid = false;
                        break;
                    }

                    //Get sub pixel values from images
                    int ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
                                          src[x + stepI] * iw10 + src[x + stepI + cn] * iw11, W_BITS1 - 5);
                    int ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                                           dsrc[dstep] * iw10 + dsrc[dstep + cn2] * iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc[dstep + 1] * iw10 +
                                           dsrc[dstep + cn2 + 1] * iw11, W_BITS1);

                    //Store values to the window buffers
                    Iptr[x] = (short) ival;
                    dIptr[0] = (short) ixval;
                    dIptr[1] = (short) iyval;

                    //Compute accum values for later gain and bias computation
                    meanI += (float) ival;
                    meanI2 += (float) (ival * ival);
                }
                if (!valid) {
                    break;
                }
            }

            if (!valid) {
                continue;
            }

            //Compute means for later gain and bias computation
            vMeanI_[level][i] = (meanI * FLT_SCALE) / winSize_.area();
            vMeanI2_[level][i] = (meanI2 * FLT_SCALE) / winSize_.area();

            Iref_[level][i] = IWinBuf.clone();
            Idref_[level][i] = derivIWinBuf.clone();
        }
    }

}

int LucasKanadeTracker::Track(const Mat &newIm, std::vector<KeyPoint> &nextPts,
                              vector<LandmarkStatus> &vMatched, const bool bInitialFlow,
                              const float minSSIM, const cv::Mat& mask) {
    //Dimensions of half of the window
    Point2f halfWin((winSize_.width - 1) * 0.5f, (winSize_.height - 1) * 0.5f);

    //covariance
    vector<cv::Mat> jac_(nextPts.size());
    for(size_t i = 0; i < jac_.size(); i++){
        jac_[i] = cv::Mat(winSize_.area(),2,CV_32F);
    }

    //Compute pyramid images
    vector<Mat> newPyr;
    cv::buildOpticalFlowPyramid(newIm, newPyr, winSize_, maxLevel_);

    const int borderGap = round(winSize_.width/2) + 1;

    //Start Lucas-Kanade optical flow algorithm
    //First iterate over pyramid levels
    for (int level = maxLevel_; level >= 0; level--) {
        const int scaleFactor = (1 << level);

        //Get images and gradients
        const Mat J = newPyr[level * 2];
        const Mat derivJ = newPyr[level * 2 + 1];

        //Buffer for fast memory access
        int j, cn = J.channels(), cn2 = cn * 2;   //cn should be 1 therefor cn2 should be 2
        AutoBuffer<short> _buf(winSize_.area() * (cn + cn2) * 2);
        int derivDepth = DataType<short>::depth;

        //Integration window buffers
        Mat IWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn), (*_buf));
        Mat JWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn), (*_buf) + winSize_.area());
        Mat derivIWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn2), (*_buf) + 2 * winSize_.area());
        Mat derivJWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn2), (*_buf) + 4 * winSize_.area());

        Mat Jvalid(winSize_,CV_8U);

        //Steps for matrix indexing
        int dstep = (int) (derivJ.step / derivJ.elemSize1());
        int stepJ = (int) (J.step / J.elemSize1());

        //Track each point at the current pyramid level
        for (int i = 0; i < prevPts_.size(); i++) {
            if(!IsUsable(vMatched[i])) continue;

            //Compute image coordinates in the reference image at the current level
            Point2f prevPt = prevPts_[i].pt * (float) (1. / (1 << level));
            //Compute image coordinates in the current frame at the current level
            Point2f nextPt;
            if (level == maxLevel_) {
                if (bInitialFlow) {
                    nextPt = nextPts[i].pt * (float) (1. / (1 << level));
                } else {
                    nextPt = prevPt;
                }
            } else {
                nextPt = nextPts[i].pt * 2.f;
            }

            nextPts[i].pt = nextPt;

            //Check that previous point and next point is inside of the
            //image boundaries
            Point2i iprevPt, inextPt;
            prevPt -= halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if (iprevPt.x < -borderGap || iprevPt.x >= derivJ.cols - borderGap ||
                iprevPt.y < -borderGap || iprevPt.y >= derivJ.rows - borderGap) {
                if (level == 0){
                    vMatched[i] = OUT_IMAGE_BOUNDARIES;
                }

                continue;
            }

            //Compute weighs for sub pixel computation
            const int W_BITS = 14, W_BITS1 = 14;
            const float FLT_SCALE = 1.f / (1 << 20);

            int x, y;
            //Compute means for later gain and bias computation
            float meanI = vMeanI_[level][i];
            float meanI2 = vMeanI2_[level][i];

            IWinBuf = Iref_[level][i].clone();
            derivIWinBuf = Idref_[level][i].clone();

            if (!IWinBuf.ptr<short>(0)) {
                if (level == 0) {
                    vMatched[i] = OUT_IMAGE_BOUNDARIES;
                }
                continue;
            }

            cv::Point2f startCoordinates = nextPt;

            //Optical flow loop
            Point2f prevDelta;
            nextPt -= halfWin;

            for (j = 0; j < maxIters_; j++) {
                //Compute weighs for sub pixel computation
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                //Check that the point is inside the image
                if (inextPt.x < -borderGap || inextPt.x >= J.cols - borderGap ||
                    inextPt.y < -borderGap || inextPt.y >= J.rows - borderGap) {
                    if (level == 0)
                        vMatched[i] = OUT_IMAGE_BOUNDARIES;
                    break;
                }

                float aJ = nextPt.x - inextPt.x;
                float bJ = nextPt.y - inextPt.y;
                int jw00 = cvRound((1.f - aJ) * (1.f - bJ) * (1 << W_BITS));
                int jw01 = cvRound(aJ * (1.f - bJ) * (1 << W_BITS));
                int jw10 = cvRound((1.f - aJ) * bJ * (1 << W_BITS));
                int jw11 = (1 << W_BITS) - jw00 - jw01 - jw10;

                //Compute alpha and beta for gain and bias
                //Compute sumI, sumI2, meanI, meanI2, Iwin, IdWin
                float meanJ = 0.f, meanJ2 = 0.f;
                int goodArea = 0;

                for (y = 0; y < winSize_.height; y++) {
                    if (vMatched[i] == OUT_IMAGE_BOUNDARIES) {
                        break;
                    }

                    //Get pointers to the images
                    const uchar *src = J.ptr() + (y + inextPt.y) * stepJ + inextPt.x * cn;
                    const short *dsrc = derivJ.ptr<short>() + (y + inextPt.y) * dstep + inextPt.x * cn2;

                    //Get pointers to the window buffers
                    short *Jptr = JWinBuf.ptr<short>(y);
                    short *dJptr = derivJWinBuf.ptr<short>(y);

                    x = 0;

                    for (; x < winSize_.width * cn; x++, dsrc += 2, dJptr += 2) {
                        //Point coordinates
                        int mx = (inextPt.x+x) * scaleFactor;
                        int my = (inextPt.y+y) * scaleFactor;

                        if(false && mask.at<uchar>(my,mx) == 0){
                            Jvalid.at<uchar>(y,x) = 0;
                        }
                        else{
                            Jvalid.at<uchar>(y,x) = 1;
                            goodArea++;
                        }

                        //Get sub pixel values from images
                        int jval = CV_DESCALE(src[x] * jw00 + src[x + cn] * jw01 +
                                              src[x + stepJ] * jw10 + src[x + stepJ + cn] * jw11, W_BITS1 - 5);
                        int jxval = CV_DESCALE(dsrc[0] * jw00 + dsrc[cn2] * jw01 +
                                               dsrc[dstep] * jw10 + dsrc[dstep + cn2] * jw11, W_BITS1);
                        int jyval = CV_DESCALE(dsrc[1] * jw00 + dsrc[cn2 + 1] * jw01 + dsrc[dstep + 1] * jw10 +
                                               dsrc[dstep + cn2 + 1] * jw11, W_BITS1);

                        //Store values to the window buffers
                        Jptr[x] = (short) jval;
                        dJptr[0] = (short) jxval;
                        dJptr[1] = (short) jyval;

                        //Compute accum values for later gain and bias computation
                        //if(mask.at<uchar>(my,mx) != 0) {
                        meanJ += (float) jval;
                        meanJ2 += (float) (jval * jval);
                        //}
                    }

                    if (vMatched[i] == OUT_IMAGE_BOUNDARIES) {
                        break;
                    }

                }

                if (vMatched[i] == OUT_IMAGE_BOUNDARIES) {
                    break;
                }

                //Compute means for later gain and bias computation
                meanJ = (meanJ * FLT_SCALE) / winSize_.area();
                meanJ2 = (meanJ2 * FLT_SCALE) / winSize_.area();

                /*meanJ = (meanJ * FLT_SCALE) / goodArea;
                meanJ2 = (meanJ2 * FLT_SCALE) / goodArea;*/

                //Compute alpha and beta
                float alpha = sqrt(meanI2 / meanJ2);
                float beta = meanI - alpha * meanJ;

                //Compute image gradient insensitive to ilumination changes
                float ib1 = 0, ib2 = 0;
                float b1, b2;
                float iA11 = 0, iA12 = 0, iA22 = 0;
                float A11, A12, A22;

                int rr = 0;

                for (y = 0; y < winSize_.height; y++) {
                    //Get pointers to the buffers
                    const short *Iptr = IWinBuf.ptr<short>(y);
                    const short *Jptr = JWinBuf.ptr<short>(y);
                    const short *dIptr = derivIWinBuf.ptr<short>(y);
                    const short *dJptr = derivJWinBuf.ptr<short>(y);

                    x = 0;
                    for (; x < winSize_.width * cn; x++, dIptr += 2, dJptr += 2) {
                        if(Jvalid.at<uchar>(y,x) == 0){
                            rr++;

                            continue;
                        }

                        int diff = Jptr[x] * alpha - Iptr[x] - beta;
                        float dx = (float) (dIptr[0] + dJptr[0] * alpha);
                        float dy = (float) (dIptr[1] + dJptr[1] * alpha);

                        ib1 += (float) (diff * dx);
                        ib2 += (float) (diff * dy);

                        iA11 += (float) (dx * dx);
                        iA22 += (float) (dy * dy);
                        iA12 += (float) (dx * dy);

                        jac_[i].at<float>(rr,0) = (float) (diff * dx);
                        jac_[i].at<float>(rr,1) = (float) (diff * dy);
                        rr++;
                    }
                }
                b1 = ib1 * FLT_SCALE;
                b2 = ib2 * FLT_SCALE;

                jac_[i] *= FLT_SCALE;

                //Compute spatial gradient matrix
                A11 = iA11 * FLT_SCALE;
                A12 = iA12 * FLT_SCALE;
                A22 = iA22 * FLT_SCALE;

                float D = A11 * A22 - A12 * A12;
                float minEig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) +
                                                      4.f * A12 * A12)) / (2 * winSize_.width * winSize_.height);

                if (minEig < minEigThreshold_ || D < FLT_EPSILON) {
                    if (level == 0)
                        vMatched[i] = BAD_FEATURE;
                    continue;
                }

                D = 1.f / D;

                //Compute optical flow
                Point2f delta((float) ((A12 * b2 - A22 * b1) * D),
                              (float) ((A12 * b1 - A11 * b2) * D));

                nextPt += delta;
                nextPts[i].pt = nextPt + halfWin;

                if (nextPts[i].pt.x < borderGap + 1 || nextPts[i].pt.x >= J.cols - 1 - borderGap ||
                    nextPts[i].pt.y < borderGap + 1|| nextPts[i].pt.y >= J.rows - 1 - borderGap) {
                    if (level == 0)
                        vMatched[i] = OUT_IMAGE_BOUNDARIES;
                    break;
                }

                if(cv::norm(nextPts[i].pt - startCoordinates) > 10){
                    nextPts[i].pt = startCoordinates;
                    if(level == 0){
                        vMatched[i] = BAD;
                    }
                    break;
                }

                if (delta.ddot(delta) <= epsilon_)
                    break;

                if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                    std::abs(delta.y + prevDelta.y) < 0.01) {
                    nextPts[i].pt -= delta * 0.5f;
                    break;
                }
                prevDelta = delta;
            }
        }
    }

    int toReturn = 0;

    const cv::Mat J = newPyr[0].clone();

    //Check outliers with SSIM
    const float C1 = (0.01 * 255)*(0.01 * 255), C2 = (0.03 * 255)*(0.03 * 255);
    const float N_inv = 1.f / (float)winSize_.area(), N_inv_1 = 1.f / (float) (winSize_.area() - 1);
    for(size_t i = 0; i < vMatched.size(); i++){
        if(IsUsable(vMatched[i])){
            if (isnan(nextPts[i].pt.x) || isnan(nextPts[i].pt.y)) {
                vMatched[i] = OUT_IMAGE_BOUNDARIES;
                continue;
            }

            float meanJ = 0.f, meanJ2 = 0.f;
            float meanI = vMeanI_[0][i], meanI2 = vMeanI2_[0][i];
            cv::Mat win = cv::Mat(winSize_,CV_16S);

            cv::Point2f nextPt = nextPts[i].pt - halfWin;
            cv::Point2i inextPt(cvFloor(nextPt.x),cvFloor(nextPt.y));

            const int W_BITS = 14, W_BITS1 = 14;
            const float FLT_SCALE = 1.f / (1 << 20);
            float aJ = nextPt.x - inextPt.x;
            float bJ = nextPt.y - inextPt.y;
            int jw00 = cvRound((1.f - aJ) * (1.f - bJ) * (1 << W_BITS));
            int jw01 = cvRound(aJ * (1.f - bJ) * (1 << W_BITS));
            int jw10 = cvRound((1.f - aJ) * bJ * (1 << W_BITS));
            int jw11 = (1 << W_BITS) - jw00 - jw01 - jw10;

            int stepJ = (int) (J.step / J.elemSize1());
            int cn = J.channels();

            int x, y;
            Mat Jvalid(winSize_,CV_8U);

            if (inextPt.x < -borderGap || inextPt.x >= J.cols - borderGap * 2 ||
                inextPt.y < -borderGap || inextPt.y >= J.rows - borderGap * 2) {
                vMatched[i] = OUT_IMAGE_BOUNDARIES;
                continue;
            }

            for (y = 0; y < winSize_.height; y++) {
                //Get pointers to the images
                const uchar *src = J.ptr() + (y + inextPt.y) * stepJ + inextPt.x * cn;

                x = 0;
                for (; x < winSize_.width * cn; x++) {
                    //Point coordinates
                    int mx = (inextPt.x+x);
                    int my = (inextPt.y+y);

                    if(!mask.empty() && mask.at<uchar>(my,mx) == 0){
                        Jvalid.at<uchar>(y,x) = 0;
                    }
                    else{
                        Jvalid.at<uchar>(y,x) = 1;
                    }

                    //Get sub pixel values from images
                    int jval = CV_DESCALE(src[x] * jw00 + src[x + cn] * jw01 +
                                          src[x + stepJ] * jw10 + src[x + stepJ + cn] * jw11, W_BITS1 - 5);

                    //Compute accum values for later gain and bias computation
                    meanJ += (float) jval;
                    meanJ2 += (float) (jval * jval);

                    win.at<short>(y,x) = (short)(jval);
                }

            }

            //Compute means for later gain and bias computation
            meanJ = (meanJ * FLT_SCALE) / winSize_.area();
            meanJ2 = (meanJ2 * FLT_SCALE) / winSize_.area();

            //Compute alpha and beta
            float alpha = sqrt(meanI2 / meanJ2);
            float beta = meanI - alpha * meanJ;

            //Correct illumination
            cv::Mat corrected = win;
            corrected /= 32;
            cv::Mat currWin;
            corrected.convertTo(currWin,CV_8U);

            cv::Mat refWin = Iref_[0][i].clone() / 32;

            //Compute means (x -> ref, y -> curr)
            float mu_x = 0.f, mu_y = 0.f;
            float sigma_x = 0.f, sigma_y = 0.f, sigma_xy = 0.f;

            for(int y = 0; y < winSize_.height; y++){
                const short* pRefWin = refWin.ptr<short>(y);
                const uchar* pCurrWin = currWin.ptr(y);

                for(int x = 0; x < winSize_.width; x++){
                    mu_x += (float) pRefWin[x];
                    mu_y += (float) pCurrWin[x];
                }
            }

            mu_x *= N_inv;
            mu_y *= N_inv;

            refWin.convertTo(refWin,CV_32F);
            currWin.convertTo(currWin,CV_32F);

            //Compute covs
            cv::Mat x_norm = refWin - mu_x;
            cv::Mat y_norm = currWin - mu_y;

            sigma_x = sqrtf(x_norm.dot(x_norm) * N_inv_1);
            sigma_y = sqrtf(y_norm.dot(y_norm) * N_inv_1);
            sigma_xy = x_norm.dot(y_norm) * N_inv_1;

            float SSIM = ((2.f * mu_x * mu_y + C1)         * (2.f * sigma_xy + C2)) /
                         ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x * sigma_x + sigma_y * sigma_y + C2));


            if(SSIM < minSSIM){
                vMatched[i] = BAD_FEATURE;
            }
            else{
                toReturn++;
            }
        }
    }


    return toReturn;
}

LucasKanadeTracker::PhotometricInformation LucasKanadeTracker::GetPhotometricInformationOfPoint(const int idx) {
    LucasKanadeTracker::PhotometricInformation photometric_information;

    for (int level = 0; level < vMeanI_.size(); level++){
        photometric_information.mean_gray_per_level.push_back(vMeanI_[level][idx]);
        photometric_information.squared_mean_gray_per_level.push_back(vMeanI2_[level][idx]);
        photometric_information.gray_reference.push_back(Iref_[level][idx]);
        photometric_information.gradient_reference.push_back(Idref_[level][idx]);
    }

    return photometric_information;
}

void
LucasKanadeTracker::InsertPhotometricInformation(cv::KeyPoint& keypoint, LucasKanadeTracker::PhotometricInformation &photometric_information) {
    prevPts_.push_back(keypoint);
    for (int level = 0; level <= maxLevel_; level++){
        vMeanI_[level].push_back(photometric_information.mean_gray_per_level[level]);
        vMeanI2_[level].push_back(photometric_information.squared_mean_gray_per_level[level]);
        Iref_[level].push_back(photometric_information.gray_reference[level].clone());
        Idref_[level].push_back(photometric_information.gradient_reference[level].clone());
    }
}

void LucasKanadeTracker::clear() {
    for(size_t level = 0; level < vMeanI_.size(); level++){
        vMeanI_[level].clear();
        vMeanI2_[level].clear();
        Iref_[level].clear();
        Idref_[level].clear();
    }

    prevPts_.clear();
}
