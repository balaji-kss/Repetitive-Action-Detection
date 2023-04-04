## Duration prediction for repetitive tasks in a video

This repository predicts the time taken for all repetitive tasks individually in an entire video. First, we predict when a particular task ends and the next task starts. Then the difference in time between two consecutive boundaries will give us the duration of that particular repetitive task.  

## Workflow
1. First we extract frames in order to load data in the data iterator fast. 
2. `ActDataset` class in `act_dataset.py` preprocesses the image and joints. We take temporal sequence([T-3, T, T + 3]) of images and joints for this case. Run `act_dataset.py` to visualize the preprocessed data.    
3. We use `ImagePoseActNet` network in `models.py` to predict task boundary.
4. Run `train.sh` to train the model
5. Run `demo.sh` to test the results on unseen videos. This script also saves all the raw boundary predictions to a lst file.
6. Run `post_process.py` to remove outliers and predict the duration of each repetitive task. This script also saves the durations in a csv file. 