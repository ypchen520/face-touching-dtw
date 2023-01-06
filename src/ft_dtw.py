from knndtw import recognizer
from knndtw import data_extraction as de
if __name__ == "__main__":
    # recognizer = recognizer.ActivityRecognizer(train_data_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data/train",
    #                                            train_act_record_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data/train-timing",
    #                                            test_data_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data/test",
    #                                            test_act_record_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data/test-timing")
    # recognizer = recognizer.ActivityRecognizer(raw_data_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/Accelerometer",
    #                                            raw_act_record_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/Start_End_Time")
    # recognizer = recognizer.ActivityRecognizer(train_data_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/mock4/mock4-train",
    #                                            train_act_record_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/mock4/mock4-train-timing",
    #                                            test_data_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/mock4/mock4-test",
    #                                            test_act_record_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/mock4/mock4-test-timing")
    # extraction = de.DataExtraction(acc_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data-entire-sequence/raw_data",
    #                                re_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data-entire-sequence/time_record",
    #                                tar_dir="/home/yupeng/FALL2020/UFII/FT-DTW/Data/data-entire-sequence/processed_data")
    # extraction.write_file(divisor=6)
    recognizer = recognizer.ActivityRecognizer(processed_data_dir="/home/yupeng/SPRING2021/FaceTouching/face-touching-dtw/data/divisor_12")
    # recognizer.cross_validation()
    # recognizer.user_dependent_test(divisor=12)
    recognizer.user_independent_test(divisor=12)
    # recognizer.visualize(index=1)
    # recognizer.recognize()
