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

#include "datasets/endomapper.h"
#include "SLAM/system.h"

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

using namespace std;

ABSL_FLAG(std::string, dataset_path, "", "Path to the video dataset");
ABSL_FLAG(std::string, settings_path, "", "Path to the settings file");
ABSL_FLAG(int, starting_frame, 0, "First frame of the dataset to process");
ABSL_FLAG(int, end_frame, 0, "Last frame of the dataset to process");

int main(int argc, char **argv) {
    // Parse command line argumemnts
    absl::ParseCommandLine(argc, argv);

    // Process command arguments
    string dataset_path = absl::GetFlag(FLAGS_dataset_path);
    if(dataset_path.empty()){
        LOG(ERROR) << "Must specify an input dataset path." << endl;
        return -1;
    }
    string settings_path = absl::GetFlag(FLAGS_settings_path);
    if(settings_path.empty()){
        LOG(ERROR) << "Must specify an input settings file." << endl;
        return -1;
    }

    int starting_frame = absl::GetFlag(FLAGS_starting_frame);
    int end_frame = absl::GetFlag(FLAGS_end_frame);

    Endomapper dataset(dataset_path);

    // Create SLAM system.
    System SLAM(settings_path);

    for (int idx = starting_frame; idx < end_frame; idx++) {
        LOG(INFO) << "Processing image " << idx;
        auto image = dataset.GetImage(idx);
        CHECK_OK(image);

        // Resize input image.
        cv::Size newSize((*image).cols/2.0f, (*image).rows/2.0f);
        cv::resize(*image, *image, newSize);

        SLAM.TrackImage(*image);
    }

    return 0;
}