# DeepFM-Keras

DeepFM written by Keras[1], similary with the tensorflow version by ChenglongChen "https://github.com/ChenglongChen/tensorflow-DeepFM"

Usage:
---
###load data and divide to train and test
dfTrain = pd.read_csv("data/train.csv")
dfTrain = dfTrain.iloc[0:int(0.7*dfTrain.shape[0]),:]
dfTest = dfTrain.iloc[int(0.7*dfTrain.shape[0]):,:]


global_columns  = dfTrain.columns.tolist()
###divide the columns by  CATEGORICAL columns	
ID_columns  = ["ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",]

qid_columns = ['id']
target_columns = ['target']


Example:
---
Folder example includes an example usage of DeepFM models for Porto Seguro's Safe Driver Prediction competition on Kaggle.

Please download the data from the competition website and put them into the example/data folder.

To train DeepFM model for this dataset, run

$ python keras_FM.py

Support:
---
Support the auc loss and log_loss as metrics




[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
