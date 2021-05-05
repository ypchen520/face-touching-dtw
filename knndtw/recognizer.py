import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from .knndtw import KnnDtw
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

isTwoLabels = True
_SEPARATED_BY_ACTIVITY = False
# _DIVISOR = 12
_INDEX_RANGE = [0]
_VISUALIZE = True

class ActivityRecognizer:
    def __init__(self, train_data_dir=None, train_act_record_dir=None,
                 test_data_dir=None, test_act_record_dir=None,
                 raw_data_dir=None, raw_act_record_dir=None,
                 processed_data_dir=None, is_entire=False):
        # classification objective
        if not isTwoLabels:
            self.labels = {1: 'MOBILE PHONE', 2: 'LYING FLAT', 3: 'COMPUTER TASKS', 4: 'WRITING', 5: 'LEISURE WALK',
                           6: 'MOVING ITEMS', 7: 'REPEATED FACE TOUCHING', 8: 'EATING & DRINKING', 9: 'SIMULATED SMOKING',
                           10:'ADJUSTING EYEGLASSES'}
        else:
            self.labels = {0: 'NON FACE TOUCHING', 1: 'FACE TOUCHING'}
        # for cross-validation
        self.is_entire = is_entire # for the case where the entire time series for an action is used as a sample
        self.processed_data = {}
        self.processed_labels = {}
        self.extract_activity_processed(processed_data_dir)
        # if raw_data_dir and raw_act_record_dir:
        #     raw_data_stream = self.make_data_stream(raw_data_dir)
            # raw_act_record_csv = self.make_act_record_csv(raw_act_record_dir)
            # '''
            # extract activities from raw data to a the raw data dictionary
            # the dictionary key would be the participant ID
            # the activities would be saved in a list
            # '''
            # self.extract_individual_activity(raw_act_record_dir, raw_data_stream, window_size, denoise_beginning, denoise_end)

        # self.train_raw_files = sorted(os.listdir(train_data_dir))
        # self.train_data = []
        # self.train_label = []
        # self.validation_data = []
        # self.validation_label = []
        # self.test_raw_files = sorted(os.listdir(test_data_dir))
        # self.test_data = []
        # self.test_label = []

        # self.act_record = pd.read_csv(act_record_csv) # read activity record
        # if train_data_dir and train_act_record_dir:
        #     train_data_stream = self.make_data_stream(train_data_dir)
        #     train_act_record_csv = self.make_act_record_csv(train_act_record_dir)
        #
        # if test_data_dir and test_act_record_dir:
        #     test_data_stream = self.make_data_stream(test_data_dir)
        #     test_act_record_csv = self.make_act_record_csv(test_act_record_dir)
        # self.extract_activity("train", train_data_stream, train_act_record_csv)
        # self.extract_activity("test", test_data_stream, test_act_record_csv)

    def select_period(self, start, end, input):
        start = pd.to_datetime(start, format="%d/%m/%Y %H:%M:%S%z")
        end = pd.to_datetime(end, format="%d/%m/%Y %H:%M:%S%z")
        # print(type(input))
        return input[(input['timestamp'] > start) & (input['timestamp'] <= end)].copy()

    # def make_data_stream(self, data_dir):
    #     raw_data_lists = []
    #     raw_files = sorted(os.listdir(data_dir))
    #     for file in raw_files:
    #         try:
    #             raw_data = pd.read_json(os.path.join(data_dir, file))
    #             raw_data_lists.append(raw_data)
    #         except UnicodeDecodeError:
    #             continue
    #     raw_data_stream = pd.concat(raw_data_lists)
    #     return raw_data_stream

    # def make_act_record_csv(self, act_record_dir):
    #     is_first_file = True
    #     act_record_files = sorted(os.listdir(act_record_dir))
    #     w_addr = os.path.join(act_record_dir, "overall.csv")
    #     w_file = open(w_addr, 'w')
    #     for file in act_record_files:
    #         f = open(os.path.join(act_record_dir, file), 'r')
    #         for line in f.readlines():
    #             if not is_first_file:
    #                 if "Start" in line:
    #                     continue
    #             is_first_file = False
    #             w_file.write(line)
    #         f.close()
    #     w_file.close()
    #     return w_addr

    def extract_activity_processed(self, data_directory):
        for file in os.listdir(data_directory):
            if file.find("final") != -1:
                pid = file[:file.index("_")]
                act_csv = os.path.join(data_directory, file)
                self.processed_data[pid], self.processed_labels[pid] = self.extract_activity_csv(act_csv)

    def extract_activity_csv(self, csv_file):
        processed_data = pd.read_csv(csv_file)
        data_list = []
        label_list = []
        data_dict = {}
        prev_index = processed_data.iloc[0, -1]
        # curr_index = processed_data.iloc[0, -1]
        label = -1
        # handle the case where the time series for an activity is not divided
        # in this case, the variable prev_index will never be updated
        if _SEPARATED_BY_ACTIVITY:
            prev_label = processed_data.iloc[0, -2]
            for i in range(processed_data.shape[0]):
                x = processed_data.iloc[i, 0]
                y = processed_data.iloc[i, 1]
                z = processed_data.iloc[i, 2]
                label = processed_data.iloc[i, -2]
                if label == prev_label:
                    if processed_data.iloc[i, -1] in _INDEX_RANGE:
                        data_dict.setdefault('accelX', []).append(x)
                        data_dict.setdefault('accelY', []).append(y)
                        data_dict.setdefault('accelZ', []).append(z)
                    # label = processed_data.iloc[i, -2]
                else:
                    print(f'logging activity: {prev_label}')
                    if isTwoLabels:
                        if prev_label <= 6:
                            prev_label = 0
                        elif 6 < prev_label <= 10:
                            prev_label = 1
                    label_list.append(prev_label)
                    data_list.append(data_dict)
                    data_dict = {}
                    data_dict.setdefault('accelX', []).append(x)
                    data_dict.setdefault('accelY', []).append(y)
                    data_dict.setdefault('accelZ', []).append(z)
                    label = processed_data.iloc[i, -2]
                prev_label = label
            # append the last sequence
            data_list.append(data_dict)
            if isTwoLabels:
                if prev_label <= 6:
                    prev_label = 0
                elif 6 < prev_label <= 10:
                    prev_label = 1
            label_list.append(prev_label)
            return data_list, label_list
        # The cases where is_entire is set to False
        print("Testing on divided time series")
        for i in range(processed_data.shape[0]):
            x = processed_data.iloc[i, 0]
            y = processed_data.iloc[i, 1]
            z = processed_data.iloc[i, 2]
            curr_index = processed_data.iloc[i, -1]
            if curr_index == prev_index:
                # if curr_index <= _INDEX_RANGE:
                data_dict.setdefault('accelX', []).append(x)
                data_dict.setdefault('accelY', []).append(y)
                data_dict.setdefault('accelZ', []).append(z)
                label = processed_data.iloc[i, -2]
                # else:
                #     print("index range exceeded")
            else:
                if isTwoLabels:
                    if label <= 6:
                        label = 0
                    elif 6 < label <= 10:
                        label = 1
                label_list.append(label)
                data_list.append(data_dict)
                data_dict = {}
                data_dict.setdefault('accelX', []).append(x)
                data_dict.setdefault('accelY', []).append(y)
                data_dict.setdefault('accelZ', []).append(z)
                label = processed_data.iloc[i, -2]
            prev_index = curr_index
        # append the last sequence
        # TODO: to be tested
        if isTwoLabels:
            if label <= 6:
                label = 0
            elif 6 < label <= 10:
                label = 1
        label_list.append(label)
        data_list.append(data_dict)
        return data_list, label_list

    # def select_inner_folds(self):
    #     self.train_data
    #     self.train_label
    #     self.validation_data
    #     self.validation_label
    #     pass



    # def extract_activity(self, data_type, data_stream, act_record_csv, window_size, denoise_beginning, denoise_end):
    #     """
    #     This function generates the following two outputs:
    #     1. a list of activities
    #     2. a list of corresponding labels
    #     Each activity is recorded using a dictionary with keys accelX, accelY, and accelZ.
    #     The activity index will match the label index.
    #
    #     For example,
    #     Assume there are ACTIVITY 1, ACTIVITY 2, and ACTIVITY 3, which correspond to
    #     the 4th, the 7th, and the 9th activities in the document, the data and label will look like:
    #
    #       train_data = [{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]
    #       train_label = [4, 7, 9, ...]
    #
    #     """
    #     # data_list = []
    #     data_list = []
    #     label_list = []
    #     act_record = pd.read_csv(act_record_csv)
    #     for i in range(act_record.shape[0]):
    #         data_dict = {}
    #         label = 0
    #
    #         # extract activity by cropping raw data
    #         # based on the activity time record
    #         # return type: pandas.core.series.Series
    #         start_time = act_record.iloc[i, 0]
    #         end_time = act_record.iloc[i, 1]
    #         # print(f"activity {act_record.iloc[i, 2]}: start: {start_time}, end: {end_time}")
    #         act = self.select_period(start_time, end_time, data_stream)
    #
    #         # extract accelerometer X, Y, and Z
    #         # use to_numpy() to convert the type from pandas.core.series.Series to numpy.ndarray
    #         data_dict.setdefault('accelX', []).extend(act['accelX'].to_numpy())
    #         data_dict.setdefault('accelY', []).extend(act['accelY'].to_numpy())
    #         data_dict.setdefault('accelZ', []).extend(act['accelZ'].to_numpy())
    #         if not isTwoLabels:
    #             label = act_record.iloc[i, 2]
    #         else:
    #             if act_record.iloc[i, 2] < 6:
    #                 label = 0
    #             elif 6 <= act_record.iloc[i, 2] <= 10:
    #                 label = 1
    #         if data_type == "train":
    #             self.train_data.append(data_dict)
    #             self.train_label.append(label)
    #         elif data_type == "test":
    #             self.test_data.append(data_dict)
    #             self.test_label.append(label)
    #         elif data_type == "raw":
    #             data_list.append(data_dict)
    #             label_list.append(label)
    #
    #     return data_list, label_list

    def extract_feature(self):
        pass

    # def visualize(self, data_type="train", index=1):
    def visualize(self, visualize_data, visualize_label, pid, data_type, divisor):
        """
        act_index: the 1-based index of the activity that will be visualized, this should not be the activity number
        """
        # visualize_data = []
        # if data_type == "train":
        #     visualize_data = self.train_data
        # elif data_type == "test":
        #     visualize_data = self.test_data
        print(f"visualizing...")
        for i in range(len(visualize_data)):
            activity_num = i // divisor + 1
            index = -1
            if data_type == "train":
                index = (i + 1) - divisor * ((i + 1) // divisor)
            elif data_type == "test":
                index = i - divisor * (i // divisor)
            filename = "_".join([str(activity_num), str(index)]) + '.png'
            x = visualize_data[i]['accelX']
            y = visualize_data[i]['accelY']
            z = visualize_data[i]['accelZ']
            plot_index = [j for j in range(len(x))]
            plot = plt.figure()
            plt.plot(plot_index, x, 'r-', label='X')
            plt.plot(plot_index, y, 'b-', label='Y')
            plt.plot(plot_index, z, 'g-', label='Z')
            plt.title('Accelerometer data for Activity #%d' % activity_num)
            plt.xlabel('index')
            plt.ylabel('Accelerometer Value')
            plt.legend()
            plt.show()
            file_path = os.path.join("results", data_type, pid)
            try:
                os.makedirs(file_path, exist_ok=True)
            except OSError:
                print(f"Directory {file_path} can not be created")
            plt.savefig(os.path.join(file_path, filename))
        # x = visualize_data[index-1]['accelX']
        # y = visualize_data[index-1]['accelY']
        # z = visualize_data[index-1]['accelZ']
        # plot_index = [i for i in range(len(x))]
        # # plot = plt.figure()
        # plt.plot(plot_index, x, 'r-', label='X')
        # plt.plot(plot_index, y, 'b-', label='Y')
        # plt.plot(plot_index, z, 'g-', label='Z')
        #
        # plt.title('Accelerometer data for Activity #%d' % visualize_label[index - 1])
        # plt.xlabel('index')
        # plt.ylabel('Accelerometer Value')
        # plt.legend()
        # # plt.show()
        # filename = activity_num + index + '.png'
        #
        # plt.savefig(os.path.join("results", data_type, pid, filename))

    # def select_outer_folds(self):
    #     self.train_data
    #     self.train_label
    #     self.test_data
    #     self.test_label
    #     return

    def calculate_accuracy(self, ground_truth, predicted):
        summation = 0
        for i in range(len(ground_truth)):
            if predicted[i] == ground_truth[i]:
                summation += 1
        return summation / len(ground_truth)

    def calculate_precision(self, ground_truth, predicted):
        tp = 0
        fp = 0
        for i in range(len(ground_truth)):
            if predicted[i] == ground_truth[i]:
                if predicted[i] == 1:
                    tp += 1
            if predicted[i] != ground_truth[i]:
                if predicted[i] == 1:
                    fp += 1
        return tp / (tp + fp)

    def calculate_recall(self, ground_truth, predicted):
        tp = 0
        fn = 0
        for i in range(len(ground_truth)):
            if predicted[i] == ground_truth[i]:
                if predicted[i] == 1:
                    tp += 1
            if predicted[i] != ground_truth[i]:
                if predicted[i] == 0:
                    fn += 1
        return tp / (tp + fn)

    def calculate_F1(self, precision, recall):
        return 2 * precision * recall / (precision + recall)

    def cross_validation(self):
        accuracy_sum = 0
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        for pid in self.processed_data.keys():
            # select 9 participants as the training data
            # select 1 participant as the testing data
            test_pid = pid
            print(f"test pid: {test_pid}")
            train_data = []
            train_label = []
            for key, value in self.processed_data.items():
                if key != test_pid:
                    train_data.extend(value)
            for key, value in self.processed_labels.items():
                if key != test_pid:
                    train_label.extend(value)
            test_data = self.processed_data[test_pid]
            test_label = self.processed_labels[test_pid] # ground truth
            print(f"ground truth: {test_label}")
            print(f"test sample size: {len(test_label)}")
            m = KnnDtw(n_neighbors=1, max_warping_window=10)
            # print(f"train label: {train_label}")
            m.fit(train_data, train_label)
            predicted_label, probability = m.predict(test_data) #predicted
            print(f"predicted label: {predicted_label}")
            print(f"test sample size sanity check: {len(predicted_label)}")
            accuracy = self.calculate_accuracy(test_label, predicted_label)
            # precision = self.calculate_precision(test_label, predicted_label)
            # recall = self.calculate_recall(test_label, predicted_label)
            # f1 = self.calculate_F1(precision, recall)
            print(f"accuracy: {accuracy}")
            # print(f"precision: {precision}")
            # print(f"recall: {recall}")
            # print(f"F1: {f1}")
            accuracy_sum += accuracy
            # precision_sum += precision
            # recall_sum += recall
            # f1_sum += f1
        average_accuracy = accuracy_sum / len(self.processed_data.keys())
        # average_precision = precision_sum / len(self.processed_data.keys())
        # average_recall = recall_sum / len(self.processed_data.keys())
        # average_f1 = f1_sum / len(self.processed_data.keys())
        print(f"average accuracy: {average_accuracy}")
        # print(f"average precision: {average_precision}")
        # print(f"average recall: {average_recall}")
        # print(f"average f1: {average_f1}")

    def user_independent_test(self, divisor=1):
        individual_result = []
        num_iteration_T = 10
        num_iteration_P = 10
        accuracy_sum_all = 0
        for test_pid in self.processed_data.keys():
            # if test_pid == '7' or test_pid == '6': # or test_pid == '8'
            #     continue
            count = 0
            accuracy_sum = 0
            print(f"test pid: {test_pid}")
            for num_participant in range(1, len(self.processed_data.keys())):
                print(f"P: {num_participant}")
                train_pids = random.sample(self.processed_data.keys(), num_participant)
                while test_pid in train_pids:
                    train_pids = random.sample(self.processed_data.keys(), num_participant)
                # print(f"training participants: {train_pids}")
                for p in range(num_iteration_P):
                    for num_train in range(1, divisor):
                        print(f"P: {num_participant}; T: {num_train}")
                        for j in range(num_iteration_T):
                            train_data = []
                            train_label = []
                            test_data = []
                            test_label = []
                            test_pid_data = self.processed_data[test_pid]
                            test_pid_label = self.processed_labels[test_pid]
                            for k in range(10):
                                # random
                                activity_index = k * divisor
                                # print(f"activity index: {activity_index}")
                                # random_indices = random.sample(range(divisor), 1 + num_train)
                                random_indices = random.sample(range(divisor), num_train)
                                # print(f"random choice: {random_indices}")
                                for i in range(len(random_indices)):
                                    if i == 0:
                                        test_data.append(test_pid_data[activity_index+random_indices[i]])
                                        test_label.append(test_pid_label[activity_index+random_indices[i]])
                                    for uid in train_pids:
                                        train_pid_data = self.processed_data[uid]
                                        train_pid_label = self.processed_labels[uid]
                                        train_data.append(train_pid_data[activity_index + random_indices[i]])
                                        train_label.append(train_pid_label[activity_index + random_indices[i]])
                                    # else:
                                    #     for uid in train_pids:
                                    #         train_pid_data = self.processed_data[uid]
                                    #         train_pid_label = self.processed_labels[uid]
                                    #         train_data.append(train_pid_data[activity_index+random_indices[i]])
                                    #         train_label.append(train_pid_label[activity_index+random_indices[i]])
                                # print(f"ground truth: {test_label}")
                                # print(f"test sample size: {len(test_label)}")
                                # print(f"train label: {train_label}")
                                # print(f"train sample size: {len(train_data)}")
                            m = KnnDtw(n_neighbors=1, max_warping_window=10)
                            m.fit(train_data, train_label)
                            predicted_label, probability = m.predict(test_data)
                            # print(f"predicted label: {predicted_label}")
                            accuracy = self.calculate_accuracy(test_label, predicted_label)
                            # print(f"accuracy: {accuracy}")
                            print(f"test pid: {test_pid}; P: {num_participant}; iteration: {p}; T: {num_train}; iteration: {j}; accuracy: {accuracy}")
                            count += 1
                            accuracy_sum += accuracy
            accuracy_individual = accuracy_sum / (num_iteration_P * (len(self.processed_data.keys())-1) * num_iteration_T * (divisor-1))
            print(count)
            print(num_iteration_P * (len(self.processed_data.keys())-1) * num_iteration_T * (divisor-1))
            print(f"accuracy for {test_pid}: {accuracy_individual}")
            individual_result.append(accuracy_individual)
            print(f"individual accuracies: {individual_result}")
            accuracy_sum_all += accuracy_individual
        average_accuracy_all = accuracy_sum_all / len(self.processed_data.keys())
        # print(f"number of participants: {len(self.processed_data.keys())}")
        print(f"overall average accuracy: {average_accuracy_all}")

    def user_dependent_test(self, divisor=1):
        individual_result = []
        num_iteration = 100
        accuracy_sum_all = 0
        # precision_sum = 0
        # recall_sum = 0
        # f1_sum = 0
        for test_pid in self.processed_data.keys():
            count = 0
            accuracy_sum = 0
            print(f"test pid: {test_pid}")
            for num_train in range(1, divisor):
                print(f"test pid: {test_pid}; T: {num_train}")
                for j in range(num_iteration):
                    train_data = []
                    train_label = []
                    test_data = []
                    test_label = []
                    current_pid_data = self.processed_data[test_pid]
                    current_pid_label = self.processed_labels[test_pid]
                    for k in range(10):
                        # random
                        activity_index = k * divisor
                        # print(f"activity index: {activity_index}")
                        random_indices = random.sample(range(divisor), 1 + num_train)
                        # print(f"random choice: {random_indices}")
                        for i in range(len(random_indices)):
                            if i == 0:
                                test_data.append(current_pid_data[activity_index+random_indices[i]])
                                test_label.append(current_pid_label[activity_index+random_indices[i]])
                            else:
                                train_data.append(current_pid_data[activity_index+random_indices[i]])
                                train_label.append(current_pid_label[activity_index+random_indices[i]])
                        # print(f"ground truth: {test_label}")
                        # print(f"test sample size: {len(test_label)}")
                        # print(f"train label: {train_label}")
                        # print(f"train sample size: {len(train_data)}")
                    m = KnnDtw(n_neighbors=1, max_warping_window=10)
                    m.fit(train_data, train_label)
                    predicted_label, probability = m.predict(test_data)
                    # print(f"predicted label: {predicted_label}")
                    # print(f"test sample size sanity check: {len(predicted_label)}")
                    accuracy = self.calculate_accuracy(test_label, predicted_label)
                    # precision = self.calculate_precision(test_label, predicted_label)
                    # recall = self.calculate_recall(test_label, predicted_label)
                    # f1 = self.calculate_F1(precision, recall)
                    print(f"test pid: {test_pid}; T: {num_train}; iteration: {j}; accuracy: {accuracy}")
                    count += 1
                    # print(f"precision: {precision}")
                    # print(f"recall: {recall}")
                    # print(f"F1: {f1}")
                    accuracy_sum += accuracy
            accuracy_individual = accuracy_sum / (num_iteration * (divisor-1))
            print(count)
            print(num_iteration * (divisor-1))
            print(f"accuracy for {test_pid}: {accuracy_individual}")
            individual_result.append(accuracy_individual)
            print(f"individual accuracies: {individual_result}")
            accuracy_sum_all += accuracy_individual
        average_accuracy_all = accuracy_sum_all / len(self.processed_data.keys())
        print(f"number of participants: {len(self.processed_data.keys())}")
        # average_precision = precision_sum / len(self.processed_data.keys())
        # average_recall = recall_sum / len(self.processed_data.keys())
        # average_f1 = f1_sum / len(self.processed_data.keys())
        print(f"overall average accuracy: {average_accuracy_all}")
        print(f"individual accuracies: {individual_result}")
        # print(f"average precision: {average_precision}")
        # print(f"average recall: {average_recall}")
        # print(f"average f1: {average_f1}")

    def recognize(self):
        m = KnnDtw(n_neighbors=5, max_warping_window=10)
        m.fit(self.train_data, self.train_label)
        print(f'training sample size: {len(self.train_label)}')
        predicted_label, probability = m.predict(self.test_data)
        print(f'ground truth: {self.test_label}')
        print(f'predicted: {predicted_label}')
        print(probability)

        report = classification_report(self.test_label, predicted_label)
        roc_auc = roc_auc_score(self.test_label, predicted_label)
        print(f'ROC AUC score: {roc_auc}')
        print(report)
        # conf_mat = confusion_matrix(predicted_label, self.test_label)
        #
        # fig = plt.figure(figsize=(10, 10))
        # width = np.shape(conf_mat)[1]
        # height = np.shape(conf_mat)[0]
        #
        # res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
        # for i, row in enumerate(conf_mat):
        #     for j, c in enumerate(row):
        #         if c > 0:
        #             plt.text(j - .2, i + .1, c, fontsize=16)
        #
        # cb = fig.colorbar(res)
        # plt.title('Confusion Matrix')
        # _ = plt.xticks(range(10), [l for l in self.labels.values()], rotation=90)
        # _ = plt.yticks(range(10), [l for l in self.labels.values()])