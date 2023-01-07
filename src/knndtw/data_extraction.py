import pandas as pd
import numpy as np
import datetime
import os
import re
import matplotlib.pyplot as plt
"""
Raw data format:
[{"accelX":"-0.30149564146995544","accelY":"0.9595218896865845","accelZ":"9.90867805480957",
"gyroA":"-51.2400016784668","gyroB":"5.740000247955322","gyroC":"-133.55999755859375",
"timestamp":"2020-10-28T16:19:26.748-04:00","id":1113181}, { ... }, ...
"""


# frequence of the accelerometer data
ACC_FREQ = 30

class DataExtraction:
    def __init__(self, acc_dir=None, re_dir=None, tar_dir=None):
        self.tar_dir = tar_dir
        '''time record'''
        self.record = []
        for f in os.listdir(re_dir):
            self.record.append(os.path.join(re_dir,f))
        '''accelerometer data'''
        acc_files = []
        self.raw_data_list = []
        for file in os.listdir(acc_dir):
            if "sensor" in file:
                acc_files.append(file)
        acc_files = sorted(acc_files)
        for file in acc_files:
            data = pd.read_json(os.path.join(acc_dir, file))
            self.raw_data_list.append(data)
        # Single dataframe
        self.raw_data = pd.concat(self.raw_data_list)
        print(f'before dropping duplicates: {self.raw_data.shape}')
        self.raw_data = self.raw_data.drop_duplicates(subset=['timestamp', 'id'])
        print(f'after dropping duplicates: {self.raw_data.shape}')
        self.raw_data.timestamp = self.raw_data.timestamp.astype(str)
        self.raw_data.timestamp = self.raw_data.timestamp.apply(lambda x: x.replace("T", " "))
        self.raw_data.timestamp = pd.to_datetime(self.raw_data['timestamp'])
        self.raw_data = self.raw_data.sort_values(by="timestamp")
        print(f'after sorting by timestamp: {self.raw_data.shape}')

    def select_period(self, start, end, input):
        start = pd.to_datetime(start, format="%d/%m/%Y %H:%M:%S%z")
        end = pd.to_datetime(end, format="%d/%m/%Y %H:%M:%S%z")
        return input[(input['timestamp'] > start) & (input['timestamp'] <= end)].copy()

    def divide_data(self, data, divisor):
        length = int(np.ceil(data.shape[0] / divisor))
        data_list = []
        if (data.shape[0] % length < length / 2) and divisor != 1:
            for i in range(divisor):
                tmp = data[i * length: (i + 1) * length].copy()
                tmp['Index'] = i
                tmp.reset_index()
                data_list.append(tmp)
        else:
            for i in range(divisor):
                if (i != divisor - 1):
                    tmp = data[i * length:(i + 1) * length].copy()
                else:
                    tmp = data[i * length:].copy()
                tmp['Index'] = i
                tmp.reset_index()
                data_list.append(tmp)
        retval = pd.concat(data_list)
        return retval.copy()

    def write_file(self, divisor=1):
        for dir in self.record:
            print(dir)
            t_rec = pd.read_csv(dir)
            data_list = []
            for i in range(t_rec.shape[0]):
                s_time = t_rec.iloc[i, 0]
                e_time = t_rec.iloc[i, 1]
                n_data = self.select_period(s_time, e_time, self.raw_data)
                n_data['ActivityNumber'] = t_rec.iloc[i, 2]
                data_list.append(n_data)
            # data per participant, with noise
            p_data = pd.concat(data_list)
            # For each activity
            act_list = []
            acts = np.unique(p_data.ActivityNumber)
            data_list = []
            for act in acts:
                tmp = p_data.query("ActivityNumber==%s" % str(act))
                tmp = tmp.iloc[20 * ACC_FREQ: -5 * ACC_FREQ]  # Remove the first 20 seconds and last 5 seconds
                data_list.append(tmp)
                retval = self.divide_data(tmp, divisor)
                act_list.append(retval)

            final_data = pd.concat(act_list) # noise removed, data divided
            p_data = pd.concat(data_list) # noise removed, not divided

            target_dir = os.path.join(self.tar_dir, "divisor_" + str(divisor))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            final_data.to_csv(os.path.join(target_dir, dir.split("/")[-1].split(".")[0] + "_final.csv"), index=False)
            p_data.to_csv(os.path.join(target_dir, dir.split("/")[-1].split(".")[0] + ".csv"), index=False)