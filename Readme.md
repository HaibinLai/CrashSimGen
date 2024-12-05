# CrashSimGen: Generating Dangerous Road Scenarios with Diffusion Models for Autonomous Driving Risk Assessment

Haibin Lai, Wenkun shen and Qihao

Southern University of Science and Technology

## Abstract

    CrashSimGen is a project that generates dangerous road scenarios using diffusion models for autonomous driving risk assessment. The project begins by encoding a large set of driving scenes as image data, and training a dangerous scenario recognizer that automatically identifies and labels potentially hazardous situations. These recognized dangerous scene thumbnails are then fed into a diffusion model to generate new dangerous scene thumbnails.

    To transform the generated thumbnails into a format usable by autonomous driving systems, we developed a parser that utilizes OpenCV image processing tools to expand the scene thumbnails into OpenDrive formatted driving scenes. These scenes are then imported into the autonomous driving simulation software Carla, where they are tested to assess and improve the vehicle's response to dangerous scenarios in a simulated environment.

## Acknowledgments

This project makes use of the following open-source code:

- [DriveSceneGen](https://github.com/SS47816/DriveSceneGen.git) by SS47816 (Licensed under Apache License 2.0).

We have integrated this code into our system, and made modifications to suit our needs to run the generative scenes in Carla. See the individual files for the list of changes.
