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

#ifndef NRSLAM_BORDERFILTER_H
#define NRSLAM_BORDERFILTER_H

#include "masking/filter.h"

/*
 * This class defines a mask over the outer borders of the image. It masks out:
 *  -rb_ and re_ rows from the beginning and the end respectively
 *  -cb_ and ce_ rows from the beginning and the end respectively
 */

class BorderFilter : public Filter {
public:
    BorderFilter(int rb, int re, int cb, int ce, int th) :
            rb_(rb),re_(re),cb_(cb),ce_(ce), th_(th) {
        filter_name_ = "BorderFilter";
    }

    cv::Mat generateMask(const cv::Mat &im);

    std::string getDescription();

private:
    int th_;
    int rb_, re_, cb_, ce_;
};


#endif //NRSLAM_BORDERFILTER_H
