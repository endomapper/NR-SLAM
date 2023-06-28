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

#include "statistics_toolbox.h"

#include <algorithm>
#include <cmath>

float Mean(const std::vector<float>& data){
    float accumulated = 0;
    for (float element : data){
        accumulated += element;
    }

    return accumulated / data.size();
}

float Median(std::vector<float>& data){
    int median_element_index = data.size() / 2;
    std::nth_element(data.begin(), data.begin() + median_element_index, data.end());

    return data[median_element_index];
}

float Sigma(const std::vector<float>& data){
    float mu = Mean(data);
    float accumulated = 0;

    for(float element : data){
        accumulated += (element - mu) * (element - mu);
    }

    return sqrtf(accumulated / data.size());
}

float chi_squared_table[30][10] {
        {0.000, 0.000, 0.001, 0.004, 0.016, 2.706, 3.841, 5.024, 6.635, 7.8792},
        {0.010, 0.020, 0.051, 0.103, 0.211, 4.605, 5.991, 7.378, 9.210, 10.5973},
        {0.072, 0.115, 0.216, 0.352, 0.584, 6.251, 7.815, 9.348, 11.345, 12.8384},
        {0.207, 0.297, 0.484, 0.711, 1.064, 7.779, 9.488, 11.143, 13.277, 14.8605},
        {0.412, 0.554, 0.831, 1.145, 1.610, 9.236, 11.070, 12.833, 15.086, 16.7506},
        {0.676, 0.872, 1.237, 1.635, 2.204, 10.645, 12.592, 14.449, 16.812, 18.5487},
        {0.989, 1.239, 1.690, 2.167, 2.833, 12.017, 14.067, 16.013, 18.475, 20.2788},
        {1.344, 1.646, 2.180, 2.733, 3.490, 13.362, 15.507, 17.535, 20.090, 21.9559},
        {1.735, 2.088, 2.700, 3.325, 4.168, 14.684, 16.919, 19.023, 21.666, 23.58910},
        {2.156, 2.558, 3.247, 3.940, 4.865, 15.987, 18.307, 20.483, 23.209, 25.18811},
        {2.603, 3.053, 3.816, 4.575, 5.578, 17.275, 19.675, 21.920, 24.725, 26.75712},
        {3.074, 3.571, 4.404, 5.226, 6.304, 18.549, 21.026, 23.337, 26.217, 28.30013},
        {3.565, 4.107, 5.009, 5.892, 7.042, 19.812, 22.362, 24.736, 27.688, 29.81914},
        {4.075, 4.660, 5.629, 6.571, 7.790, 21.064, 23.685, 26.119, 29.141, 31.31915},
        {4.601, 5.229, 6.262, 7.261, 8.547, 22.307, 24.996, 27.488, 30.578, 32.80116},
        {5.142, 5.812, 6.908, 7.962, 9.312, 23.542, 26.296, 28.845, 32.000, 34.26717},
        {5.697, 6.408, 7.564, 8.672, 10.085, 24.769, 27.587, 30.191, 33.409, 35.71818},
        {6.265, 7.015, 8.231, 9.390, 10.865, 25.989, 28.869, 31.526, 34.805, 37.15619},
        {6.844, 7.633, 8.907, 10.117, 11.651, 27.204, 30.144, 32.852, 36.191, 38.58220},
        {7.434, 8.260, 9.591, 10.851, 12.443, 28.412, 31.410, 34.170, 37.566, 39.99721},
        {8.034, 8.897, 10.283, 11.591, 13.240, 29.615, 32.671, 35.479, 38.932, 41.40122},
        {8.643, 9.542, 10.982, 12.338, 14.041, 30.813, 33.924, 36.781, 40.289, 42.79623},
        {9.260, 10.196, 11.689, 13.091, 14.848, 32.007, 35.172, 38.076, 41.638, 44.18124},
        {9.886, 10.856, 12.401, 13.848, 15.659, 33.196, 36.415, 39.364, 42.980, 45.55925},
        {10.520, 11.524, 13.120, 14.611, 16.473, 34.382, 37.652, 40.646, 44.314, 46.92826},
        {11.160, 12.198, 13.844, 15.379, 17.292, 35.563, 38.885, 41.923, 45.642, 48.29027},
        {11.808, 12.879, 14.573, 16.151, 18.114, 36.741, 40.113, 43.195, 46.963, 49.64528},
        {12.461, 13.565, 15.308, 16.928, 18.939, 37.916, 41.337, 44.461, 48.278, 50.99329},
        {13.121, 14.256, 16.047, 17.708, 19.768, 39.087, 42.557, 45.722, 49.588, 52.33630},
        {13.787, 14.953, 16.791, 18.493, 20.599, 40.256, 43.773, 46.979, 50.892, 53.672}
};

absl::flat_hash_map<float, int> alpha_to_idx = {{0.995, 0}, {0.99, 1}, {0.975, 2}, {0.95, 3}, {0.9, 4},
                                                {0.1, 5}, {0.05, 6}, {0.025, 7}, {0.01, 8}, {0.005, 9}};


float ChiSquared(const int degrees_of_freedom, const float alpha) {
    return chi_squared_table[degrees_of_freedom][alpha_to_idx[alpha]];
}