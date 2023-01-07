## Entry Point

* ft_dtw.py

## Data

- We implemented a monitoring [application](https://github.com/ypchen520/mini-ROAMM) that records accelerometer data at a rate of **30Hz** on the **smartwatch** using Tizen Studio
- We recruited 10 participants, including 5 females and 5 males who ranged in age from 20 to 83 (M = 47.7, SD = 27.7)
- we selected 10 everyday activities
  - Non-face-touching
    1. Using a mobile phone
    2. Lying flat on the back
    3. Computer tasks
    4. Writing
    5. Leisurely walk
    6. Moving items from one location to another
  - Face-touching
    1. Repeated face touching
    2. Eating and drinking
    3. Simulated smoking
    4. Adjusting eyeglasses
- We accumulated ten streams of data for each of the ten participants, with each stream representing a 3-minute task consisting of consecutive repetitions of an activity. 
  - Therefore, our dataset contains **100 time series that are each 3 minutes long**

### Data Preprocessing

- We applied data preprocessing techniques to each **3-minute time series** to 
  1. capture a stream of cyclical activity repetitions without the irregular noise at the start and end of the stream, and then 
    - We examined the visualization of the time series to identify the extent of irregular noise at the start and end of each task caused by the starting and ending of the task by the user; 
    - based on this observation, we removed **the first 20 and last 5 seconds** from all samples
    - the ```visualize()``` method is implemented in [the ```recognizer``` module](#implementation-the-recognizer-module)
  2. divide the stream into smaller segments, each representing a repetition of an activity, to be used for classification
    - We conducted preliminary experiments to select the optimum number of segments (e.g., 1, 12, 18, 20, 30, 36), and found that **dividing each time series into 12 segments** of roughly **15 seconds** in length resulted in the best recognition performance. 
- Our resulting dataset contains a total of **1200 samples** = 10 (participants) × 10 (activities) × 12 (segments).
- This repository only includes the preprocessed dataset containing 1200 time series, each of roughly **15 seconds** in length
  - If you are interested in the original raw dataset, please reach out to me at <yupengchen@ufl.edu>

### Implementation: The ```data_extraction``` Module

- This module defines the ```DataExtraction``` class
  - we instantiate a ```DataExtraction``` object with the following arguments
    - ```acc_dir```: the raw data directory
    - ```re_dir```: the timer records directory
    - ```tar_dir```: the target directory for the preprocessed data
  - ```write_file()```
    - input
      - ```divisor```: the number that will be used to divide a 3-minute stream into smaller segments, each representing a repetition of an activity
    - outputs 
      - **two files** for each activity (a total of 20 ```.csv``` files) in the "divisor_[DIVISOR]/" folder
        - the file name is based on the ID of the participant
        - "[ID].csv": noise removed, not divided
        - "[ID]_final.csv": noise removed, data divided

## The DTW-Based Classifier

- DTW is a dynamic programming algorithm that is capable of measuring the similarity between two temporal sequences of different lengths. 
  - Berndt and Clifford provide a more detailed explanation in their [work](https://www.aaai.org/Papers/Workshops/1994/WS-94-03/WS94-03-031.pdf).
- In our experiment, we collected accelerometer data for **the three spatial dimensions**
  - a sample represents a repetition of an activity consisting of three time series, each corresponding to the three axes: ```x```, ```y```, and ```z```
- We calculate the matching between two samples A and B using the following equation: 

$$ DTW(A,B) = \sqrt{DTW(A_x,B_x)^2 + DTW(A_y,B_y)^2 + DTW(A_z,B_z)^2} $$

- In this equation, we used the open-source Python package [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-distance-measure-between-two-time-series) to calculate an overall DTW distance between the two samples based on the three DTW distance values in the x, y, and z directions.
- To classify a test sample, we first calculated the DTW distance between the test sample and all training templates individually, and then assigned to the test sample the label of the training template that resulted in the minimum DTW distance

### Implementation: The ```knndtw``` Module

- This module defines the ```KnnDtw``` class
  - [```fit()```](./knndtw/knndtw.py#L34)
    - Assign training data to ```self.x```
      - ```self.x``` is a list of dictionaries with each containing a repitition of an activity: 
        - ```[{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]```
      - Each dictionary in the ```self.x``` list consists of three time series as lists, each corresponding to the three axes: ```accelX```, ```accelY```, and ```accelZ```
        -  ```{ACTIVITY N}: {'accelX': [...], 'accelY': [...], 'accelZ': [...]}```
    - Assign training labels to ```self.label```
  * dtw_distance
    * input
      * ts_a, ts_b: array of shape [n_samples, n_timepoints]
      * Two arrays containing n_samples of timeseries data whose DTW distance between each sample of A and B will be compared
    * output
      * Returns the DTW similarity distance between two **2-D time series numpy arrays**.
  * dist_matrix
    * input
      * x: training data presented as a list dictionaries
        * [{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]
      * y: testing data presented as a list dictionaries
        * [{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]
  * predict
    * input
      * x: testing data presented as a list dictionaries
        * [{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]
    * process
      * calculate the distance matrix

## Analysis

- We formulated the problem in two ways: **binary classification** and **multiclass classification** as shown below
  ```Python
  # classification objective
  if not _BINARY_CLASSIFICATION:
      self.labels = {1: 'MOBILE PHONE', 2: 'LYING FLAT', 3: 'COMPUTER TASKS', 4: 'WRITING', 5: 'LEISURE WALK',
                    6: 'MOVING ITEMS', 7: 'REPEATED FACE TOUCHING', 8: 'EATING & DRINKING', 9: 'SIMULATED SMOKING',
                    10:'ADJUSTING EYEGLASSES'}
  else:
      self.labels = {0: 'NON FACE TOUCHING', 1: 'FACE TOUCHING'}
  ```
- We aimed at training and testing our DTW-based classifier with all possible combinations of training and test samples in both **user-dependent** and **user-independent**
scenarios
- User-dependent scenario
  - The classifier was trained and tested on a specific user (i.e., best-case accuracy) using cross-validation.
  - For each participant, for each activity type, $T$ samples were randomly selected as the training data and one additional sample was selected for testing. 
    - We repeated this process 100 times for each value of $T$ (e.g., 1 to 11) and calculated the average of the results for each participant. 
    - The results from the $10(\text{participants}) \times 10(\text{activities}) \times 100(\text{iterations for T}) \times 11(\text{T}) = 1.1 \times 10^5$ classification tests were averaged into an overall classification accuracy
- User-independent scenario
  - We also used cross-validation to examine how well a classifier can **generalize** to activities performed by users whose data are not included in the training set.
  - We selected one participant’s data to use for testing and randomly selected $P$ other participants’ data to use for training.
    - #training participant: $P \text{(1 to 9)}$ 
    - #test participant: $1$
  - For each activity type, $T$ samples were randomly selected from each training participant while one sample was selected from the testing participant. 
    - #training samples per training participant: $T \text{(1 to 11)}$
      - #total training samples for this classification test: $P \text{(1 to 9)} \times T \text{(1 to 11)}$
    - #test sample from the testing participant: $1$
  - We repeated the process 10 times for each $P$ (e.g., 1 to 9) and 10 times for each $T$ (e.g., 1 to 11). 
    - The results from the $10(\text{participants}) \times 10(\text{activities}) \times 10(\text{iterations for P}) \times 9(\text{P}) \times 10(\text{iterations for T}) \times 11(\text{T}) = 9.9 \times 10^5$ classification tests were averaged into an overall classification accuracy

### Implementation: The ```recognizer``` Module

- The analyses we conducted are implemented in the ```recognizer``` Python module
- This module defines the ```ActivityRecognizer``` class
  - [```__init__()```](./knndtw/recognizer.py#L17)
    * directories
      * preprocessed raw data points
      * labels (recordings)
      * self.processed_data = {}
        * PID : [a list of time series]
          * a time series is presented by a dictionary (for the purpose of tri-axial data)
          * this dictionary represents a subsuquence extracted from a 3-minute activity time series
      * self.processed_labels = {}
        * PID : [a list of labels corresponding to each time series]
  * extract_activity_processed
    * input
      * processed data directory
    * process
      * call extract_activity_csv to extract data from csv files in each folder that contains preprocessed data that consists of time series generated by dividing the original entire activity time series (3-minute long)
      * extracts PID based on the file name
      * assign a list of data to self.data[PID]
      * assign a list of labels to self.labels[PID]
    * output
      * NA
  * extract_activity_processed
    * input
      * csv file that contains the processed data
    * process
      * use pd.read_csv()
      * index
        * since each 3-minute activity time series is divided into multiple smaller time series, each subsequence has an index in the csv file
    * output
      * a list of dictionaries
        * each dictionary consists of: {'accelX': [subsequence], 'accelY': [...], 'accelZ': [...]}
        * e.g. [{}, {}, {}, ..., {}, {}, ...]
      * a list of labels
        * each label corresponds to a dictionary (a subsequence)
        * e.g. [1, 1, 1, ..., 6, 6, ...]
  * cross_validation
    * input
      * csv file that contains the processed data
    * process
      * training data []
        * PIDs: self.processed_data.keys()
          * select one as the test PID
        * the suquences (presented by a dictionary) from the rest PIDs are all appended to train_data
      * training labels []
      * test data []
        * self.processed_data[test_pid]
      * test labels []
      * KnnDtw.fit(train_data, train_label)
      * KnnDtw.predict(test_data)
        * get predicted label as a list
    * output
      * print average_accuracy
      * print average_precision
      * print average_recall
      * print average_f1