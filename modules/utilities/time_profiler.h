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

#ifndef NRSLAM_TIME_PROFILER_H
#define NRSLAM_TIME_PROFILER_H

#include <chrono>
#include <functional>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"

class TimeProfiler {
public:
    TimeProfiler() {}

    template<class Function>
    void MeasureExectutionTimeOfFunction(const std::string& identifier,
                                         Function&& function) {
        std::chrono::high_resolution_clock::time_point time_begin =
                std::chrono::high_resolution_clock::now();
        function();
        std::chrono::high_resolution_clock::time_point time_end =
                std::chrono::high_resolution_clock::now();

        std::chrono::duration <float, std::milli> time_ms =
                time_end - time_begin;

        watchers_[identifier].push_back(time_ms.count());
        LOG(INFO) << time_ms.count();
    }

    void Tic(std::string identifier);

    void Toc(std::string identifier);

    void PrintStatisticsForIdentifier(const std::string& identifier) const;

    void SaveStatisticsToFile(const std::string& file_name);

private:
    absl::flat_hash_map<std::string, std::vector<float>> watchers_;

    typedef std::chrono::steady_clock::time_point time_point;
    absl::flat_hash_map<std::string, time_point> initial_measurements_;
};



#endif //NRSLAM_TIME_PROFILER_H
