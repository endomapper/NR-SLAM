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

#include <fstream>
#include "time_profiler.h"

#include "statistics_toolbox.h"

#include "absl/log/check.h"
#include "absl/log/log.h"

void TimeProfiler::Tic(std::string identifier) {
    initial_measurements_[identifier] = std::chrono::steady_clock::now();
}

void TimeProfiler::Toc(std::string identifier) {
    time_point end = std::chrono::steady_clock::now();
    float timeElapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - initial_measurements_[identifier]).count();
    watchers_[identifier].push_back(timeElapsed);
}

void TimeProfiler::PrintStatisticsForIdentifier(const std::string &identifier) const {
    CHECK(watchers_.contains(identifier));

    float mean = Mean(watchers_.at(identifier));

    LOG(INFO).NoPrefix() << "Time statistics for identifier \"" << identifier << "\":";
    LOG(INFO).NoPrefix() << "\t-Number of samples: " << watchers_.at(identifier).size();
    LOG(INFO).NoPrefix() << "\t-Mean (ms): " << mean;
}

void TimeProfiler::SaveStatisticsToFile(const std::string &file_name) {
    std::ofstream time_data_writer(file_name);

    for (const auto &[identifier, data] : watchers_) {
        time_data_writer << identifier << ": " << Mean(watchers_.at(identifier)) << " " << Sigma(watchers_.at(identifier)) << " ";

        for (auto d : data) {
            time_data_writer << " " << d;
        }

        time_data_writer << std::endl;
    }

    time_data_writer.close();
}
