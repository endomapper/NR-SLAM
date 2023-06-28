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

#ifndef NRSLAM_MASKER_H
#define NRSLAM_MASKER_H

#include "filter.h"

#include "absl/container/flat_hash_map.h"

/*
 * This class collects a set of filter to generate a global mask from all of them
 */

class Masker {
public:
    Masker(){};

    /*
     * Load filters from a .txt file with the following format:
     * <filterName> <param_1> <param_2>
     */
    void loadFromTxt(std::string path);

    /*
     * Adds a filter to the masker
     */
    void addFilter(std::unique_ptr<Filter>& f);

    /*
     * Removes the filter at pos idx
     */
    void deleteFilter(size_t idx);

    /*
     * Applies all filters stored and generates a global mask
     */
    cv::Mat mask(const cv::Mat& im);

    // Applies all filters and returns all the masked generated and the global
    // obtained from applying all of them at the same time.
    absl::flat_hash_map<std::string, cv::Mat> GetAllMasks(const cv::Mat& im);

    std::string printFilters();

private:
    std::vector<std::unique_ptr<Filter>> filters_;
};


#endif //NRSLAM_MASKER_H
