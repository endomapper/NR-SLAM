# NR-SLAM: Non-Rigid Monocular SLAM

### V0.1, June 28th, 2023
**Authors:** [Juan J. Gómez Rodríguez](https://jj-gomez.github.io/), [José M. M. Montiel](http://webdiis.unizar.es/~josemari/), [Juan D. Tardós](http://webdiis.unizar.es/~jdtardos/).

NR-SLAM is a novel monocular deformable SLAM system founded on the combination of a **Dynamic Deformation Graph** with a **Visco-Elastic deformation model**.
It is able to reconstruct medical imagery with surfaces with different types of topologies and deformations and can use **pinhole** and **fisheye** cameras.

We provide examples to run NR-SLAM in the Hamlyn and in the Endomapper datasets. Videos of some example executions can be found [here](https://drive.google.com/file/d/12KNHVLE05uoO4x9eZ-qHlGtQ-JPZaAnD).

<a href="https://youtu.be/N-N0ugRjR2s" target="_blank"><img src="https://youtu.be/N-N0ugRjR2s/0.jpg"
alt="NR-SLAM" width="240" height="180" border="10" /></a>



### Related Publications:
[NR-SLAM] Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **NR-SLAM: Non-Rigid Monocular SLAM**, *ArXivxxx.yyy*. **[PDF](TO BE UPLOADED)**.

[Deformable tracking] Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **Tracking monocular camera pose and deformation for SLAM inside the human body**, *IEEE/RSJ International Conference on Intelligent Robots and Systems 2022*. **[PDF](https://arxiv.org/abs/2204.08309)**.

# 1. License

NR-SLAM is released under [AGPL license](https://github.com/endomapper/NR-SLAM/LICENSE). For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/endomapper/NR-SLAM/Dependencies.md).

For a closed-source version of NR-SLAM for commercial purposes, please contact the authors: jjgomez (at) unizar (dot) es, josemari (at) unizar (dot) es, tardos (at) unizar (dot) es.

If you use NR-SLAM in an academic work, please cite:

    @article{NR-SLAM,
      title={{NR-SLAM}: Non-Rigid Monocular {SLAM}},
      author={G\´omez, Juan J. AND Montiel, 
              Jos\'e M. M. AND Tard\'os, Juan D.},
      journal={ArXiV xxx.yyy},
      year={2023}
     }

# 2. Prerequisites
We have tested the library in **Ubuntu 20.04.4 LTS** but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure good performance and provide more stable and accurate results.

## C++17
We use several functionalities of C++17.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 3.0. Tested with OpenCV 3.2.0 and 4.4.0**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## Boost
We use [Boost](https://www.boost.org/) for directory operations.

## MLPACK
We use [MLPACK](https://www.mlpack.org/) for clustering operations.

# 3. Building NR-SLAM library and examples

Clone the repository:
```
git clone https://github.com/endomapper/NR-SLAM NR_SLAM
```

We provide a script `build.sh` to build the *third_party* libraries and *NR-SLAM*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd NR-SLAM
chmod +x build.sh
./build.sh
```

This will create **libNR-SLAM_d**  at *build/lib* folder and the executables in *build/bin* folder.

# 4. Endomapper Examples
[Endomapper dataset](https://www.synapse.org/#!Synapse:syn26707219/wiki/615178) is composed by a set of real and
simulated colonoscopies from a monocular endoscope. we provide an example program to launch the sequence for
this dataset, both the real and simulated videos.

1. Grab your Endomapper video from https://www.synapse.org/#!Synapse:syn26707219/wiki/615178

2. Execute the following command for a real colonoscopy sequence:
```
./build/bin/endomapper --dataset_path <video_path> 
                       --settings_path .data/endomapper/settings.yaml 
                       --starting_frame <starting_frame> 
                       --end_frame <last_frame>
```

3. Execute the following command for a simulated colonoscopy sequence:
```
./build/bin/simulation --dataset_path <dataset_folder> 
                       --settings_path .data/simulation/settings.yaml 
                       --starting_frame <starting_frame> 
                       --end_frame <last_frame>
```

# 5. Hamlyn Examples
[Hamlyn dataset](http://hamlyn.doc.ic.ac.uk/vision/) is a set of endoscopy sequences recorded with a monocular and stereo endoscope. 

1. Download the dataset from the [webpage](http://hamlyn.doc.ic.ac.uk/vision/). (Disclaimer: unfortunately the web page is often down. Please contact the dataset authors to get the images).

2. Execute the following command:
```
./build/bin/hamlyn --dataset_path <video_path> 
                       --settings_path .data/hamlyn_<i>/settings.yaml 
                       --starting_frame <starting_frame> 
                       --end_frame <last_frame>
```
