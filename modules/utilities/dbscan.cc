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

#include "dbscan.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>

using namespace mlpack;
using namespace mlpack::util;

using namespace std;

std::vector<int> Dbscan2D(std::vector<cv::Point2f>& points){
    mlpack::DBSCAN<RangeSearch<>,OrderedPointSelection>
            clustering(0.2,3);

    arma::mat data(2, points.size());
    arma::vec norms(points.size());
    arma::Row<size_t> labels(points.size());
    for(size_t i = 0; i < points.size(); i++){
        data(0,i) = points[i].x;
        data(1,i) = points[i].y;

        norms(i) = arma::norm(data.col(i));
    }

    // Normalize data.
    float maxNorm = arma::max(norms);
    float minNorm = arma::min(norms);
    arma::vec normsNormalized = (norms - minNorm) / (maxNorm - minNorm);
    data.each_row() /= norms.t();
    data.each_row() %= (normsNormalized.t() + 0.1f);

    clustering.Cluster(data,labels);

    vector<int> vLabels(points.size());
    for(size_t i = 0; i < points.size(); i++){
        vLabels[i] = (labels(i) == SIZE_MAX) ? -1 : labels(i);
    }

    return vLabels;
}

std::vector<int> Dbscan3D(std::vector<Eigen::Vector3f>& points){
    mlpack::DBSCAN<RangeSearch<>,OrderedPointSelection>
            clustering(2.5, 5); // (4.5, 5) for Hamlyn 20, (2.5, 5) for Hamlyn 21

    arma::mat data(3, points.size());
    arma::Row<size_t> labels(points.size());
    for(size_t i = 0; i < points.size(); i++){
        data(0,i) = points[i].x();
        data(1,i) = points[i].y();
        data(2,i) = points[i].z();
    }

    clustering.Cluster(data,labels);

    vector<int> vLabels(points.size());
    for(size_t i = 0; i < points.size(); i++){
        vLabels[i] = (labels(i) == SIZE_MAX) ? -1 : labels(i);
    }

    //Sort cluster IDs
    map<int,int> mClusterSize;
    for(size_t i = 0; i < vLabels.size(); i++){
        mClusterSize[vLabels[i]]++;
    }

    vector<pair<int,int>> vOrderedClusters(mClusterSize.begin(),mClusterSize.end());

    auto comparator = [](pair<int,int> p1, pair<int,int> p2){
        return (p1.second > p2.second) || (p1.second == p2.second && p1.first < p2.second);
    };

    sort(vOrderedClusters.begin(),vOrderedClusters.end(),comparator);

    map<int,int> mTranslation;
    for(size_t i = 0; i < vOrderedClusters.size(); i++){
        mTranslation[vOrderedClusters[i].first] = i;
    }

    for(size_t i = 0; i < vLabels.size(); i++){
        vLabels[i] = mTranslation[vLabels[i]];
    }

    return vLabels;
}

std::vector<int> DbscanND(std::vector<Eigen::VectorXf>& points){
    const float epsilon = 0.1 * points[0].size();
    mlpack::DBSCAN<RangeSearch<>,OrderedPointSelection>
            clustering(epsilon, 10);

    arma::mat data(points[0].size(), points.size());
    arma::vec norms(points.size());
    arma::Row<size_t> labels(points.size());
    for (int col = 0; col < points.size(); col++) {
        for (int row = 0;  row < points[col].size(); row++) {
            data(row, col) = points[col](row);
        }

        norms(col) = arma::norm(data.col(col));
    }


    clustering.Cluster(data,labels);

    vector<int> vLabels(points.size());
    for(size_t i = 0; i < points.size(); i++){
        vLabels[i] = (labels(i) == SIZE_MAX) ? -1 : labels(i);
    }

    return vLabels;
}
