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

#ifndef NRSLAM_FILTER_H
#define NRSLAM_FILTER_H

#include<opencv2/core/core.hpp>

/*
 * This class defines a generic filter to generate a single mask from an input image
 */

class Filter {
public:
    Filter(){};

    /*
     * Virtual method for generating the mask
     */
    virtual cv::Mat generateMask(const cv::Mat &im) = 0;

    /*
     * Retrieves a short description of the filter
     */
    virtual std::string getDescription() = 0;

    std::string GetFilterName() {return filter_name_;};

protected:
    std::string filter_name_;
    std::string filterDescription;
};


#endif //NRSLAM_FILTER_H
