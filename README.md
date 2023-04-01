# Introduction

TuMeke's main goal as a company is to reduce injuries in manufacturing settings. The leading cause of bodily injury in manual work is hazardous postures repeated for a long period of time. Therefore, a very natural question to ask is: how often is a particular task repeated? Motions/tasks that are repeated frequently are likely to cause injury/discomfort. This task centers around estimating the frequency of repetition of a manual labor task given its video footage. Ultimately, you'll need to estimate how long each cycle of a job is, which gives rise to its frequency. For example, if we were to consider the short clip `sample.mp4` in the root folder of this directory, you'll notice that the gentlemen carries a box back and forth from the conveyor belt every 6 seconds or so. If we were to write down the times that each cycle takes, we might get an output like:

6
7
6.5
8
....

We want to output a similar list of cycle length times for arbitrary videos. 

# Scenario

The scenario for this assignment is simple: We provide a short clip of a task with ground truth cycle times -- you can think of this as a manufacturing manager manually labeling cycle lengths for a short video as a sample to our system. Then, we provide additional clips of the same job that you need to predict cycle times for, using the data in the shorter clip to guide your model. 

# Data

The first folder, `simple_data`, contains three sets of videos, each approximately an hour long split into multiple clips. These videos were taken about 2 years ago as we were starting the company, so please excuse the somewhat cluttered/unprofessional background you see :) The first video is of our cofounder deadlifting a weight. You will see ground truth cycle times for "clip_1", but not "clip_2" or "clip_3". Each subsequent video also has ground truth cycle times for the first one or two clips, but not the rest. 

Each folder also has a few files. `video.mp4` is the raw video file. `2d.csv` contains joint data of the video. There are 15 joints tracked (see the header of the CSV for joint order). These data were produced from a neural network, so they may be noisy. We have already discarded all low confidence (< .35) predictions from the network. If data is missing for a particular frame, the `*.csv` file will contain `nan` for that joint. 

The second folder `hard_data`, is structured very similarly to the first, but contains more challenging examples. These are more real-world samples where the jobs are less structured and the amount of ground truth data is limited. For example, in the previous set, `lifting_1` contains 20 minutes of cycle time data, whereas in `hard_data`, there is only around 2 mins available for both samples. This is more realistic with regards to the data we'd get from customers.

## `simple_data/lifting_1/clip_1` ground truth explanation

To further illustrate how the data is structured, here's a short explanation of the GT times for the `simple_data/lifting_1/clip_1` video. The ground truth times in `simple_data/lifting_1/clip_1` represent the length of time each cycle of the deadlift takes (in seconds). For example, the first few numbers in the ground truth file are: 30.528, 34.265, 32.074, 35.447. This means the first cycle took 30.5s, the second cycle took 34.2s, so on and so forth. If you were to go to the (30.528 + 34.265 + 32.074) = 96.867s timestamp in the video (located at `simple_data/lifting_1/clip_1/video.mp4`), you should see the start of the 4th cycle. 

# Cycle interpretations

Each video contains some kind of cyclical task. Here we're including some verbal descriptions of what the cycles are, to help you make sense of the ground truth data. Note that the model should try and infer what the cycle means from the ground truth provided (and project that forward to the other data for the same video), but we're including this here to help you understand what's going on the video:

- `simple_data/lifting_1`: This is a video of one of our cofounders deadlifting. The cycle begins when he racks the weight off of the stool, and ends when he sets the weight back on the stool
- `simple_data/lifting_2`: This is a video of one of our cofounders lifting/lowering a box. The cycle begins when he lifts the box off the ground, and ends when he sets the items in the box back inside (after lowering the box to the floor)
- `simple_data/lifting_3`: Same as `simple_data/lifting_2`
- `hard_data/kontoor`: This is a video of someone packing clothes into a box. The cycle begins when she reaches for apparel to the right of her, and ends when she sets the apparel into the loading box.
- `hard_data/folding`: This is a video of one of our cofounders folding laundry. The cycle begins when he reaches for a item of clothing, and ends when he sets the folded item down.

# Input/output

The input to your model should be the joint data/video data (whatever combo you'd like to use), and the output should a list of durations for each cycle of motion that's repeated in the job. You can use the available ground truth times to "train" your system. 

# Code

- `pose_dataset.py` contains some sample code demonstrating how to load in data as a pytorch `Dataset`. Note that the current method of loading frames from the video in `pose_dataset.py` is fairly ineffecient and should be modified somehow (perhaps by breaking up the video into frames beforehand) if you want to use video data for your model.
- `main.py` demonstrates a simple example loading the dataset with a pytorch `DatasetLoader`. 
- `requirements.txt` contains a small list of python deps for `pose_dataset.py` and `main.py`.

# Getting Started

To get started, we recommend taking a look at the input video footage. Use the grouth truth times to get a sense for what the cycles visually look like. Remember that any sort of technique (signal processing, neural networks, classical ML) is fair game for a solution. 

# Deliverables

Alongside your code, please include a short writeup explaining your approach -- some type of visuals (charts/diagrams) are always helpful to communicate your solution/results. Feel free to reach out to me (diwakar@tumeke.io) if you have any questions about the project!

# Logging into AWS

You have access to a T4 GPU on AWS in case you'd like to use accelerated computing for this project. In an email you should have received a `*.pem` file that contains your credentials to access the machine, and the url of the machine. Place the `*.pem` file in your `~/.ssh/` directory, and then you can sign in with: `ssh -i ~/.ssh/<PEM_FILENAME> ubuntu@<AWS_URL`. The machine already should have most common ML dependencies installed (NVIDIA drivers, pytorch, cuda, etc). Use `conda info --envs` to see a list of all the conda environments on the remote machine. The `pytorch_latest_p37` environment has all pytorch deps loaded with python 3.7. 
