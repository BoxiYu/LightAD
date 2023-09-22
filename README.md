# LightAD
A toolkit for Light Log Anomaly Detection [ICSE'24]

## Step 1: Check Python Dependencies

To install LightAD dependencies, please run:

```shell
pip install -r requirements.txt
```

## Step 2: Prepare Datasets

The example 100k HDFS dataset is under ```/datasets/orignal_datasets```

The original full datasets can be found at: (1) HDFS dataset: https://doi.org/10.5281/zenodo.1144100, (2) Supercomputer datasets: https://www.usenix.org/cfdr-data.

If you want to run LightAD on the full datasets, you should download the data from the above websites, put the corresponding files in ```/datasets/orignal_datasets``` and name them by the names of the datasets without any suffix.

For HDFS dataset, you don't have worry about the ```anamaly_label.csv```. The file contains all the labels for the full dataset.

## Step 3: Preprocess Datasets
To preprocess any dataset, please run:

```shell
python preprocess.py --dataset [dataset_you_want_to_preprocess] 
```
The [dataset_you_want_to_preprocess] can be ```hdfs```, ```bgl```, ```spirit```, ```liberty```, ```tbird```.
## Step 4: Conduct Log Anomaly Detection
### On the HDFS Dataset:

If you want to conduct anomaly detection on the entire HDFS dataset, please run:

```shell
python main_hdfs.py --model [model_you_want_to_use]
```
If you want to conduct anomaly detection on the deduplicated HDFS dataset, please run:

```shell
python main_hdfs.py --model [model_you_want_to_use] --eliminate True
```

The models that can be deployed on HDFS are ```"knn"``` (K-Nearest-Neighbor), ```"dt"``` (Decision Tree), and ```"slfn"``` (Single Layer Feed Forward Neural Network).
### On the Supercomputer Datasets:

If you want to conduct anomaly detection on the supercomputer datasets, please run:

```shell
python main_super.py --dataset [dataset_you_want_to_use]
```
The supercomputer datasets [dataset_you_want_to_use] can be ```"bgl"```, ```"tbird"```, ```"spirit"```, and ```"liberty"```. Only ```"knn"``` is leveraged for we do not preprocess supercomputer datasets into numerical vectors.
## Step 5: Select the Optimal Model

This step is performed on the deduplicated HDFS dataset which can be obtained by:
```shell
python preprocess.py --dataset hdfs --eliminate True 
```
If you want to get the ModelScore of a model (the higher the ModelScore, the better the model performs under the current optimization strategy), please run:

```shell
python main_opt.py --model [model_you_want_to_use] --l1 [importance_of_model_accuracy] --l2 [importance_of_train_time] --l3 [importance_of_infer_time]
```
```l1```, ```l2```, and ```l3``` respectively represent the relative importance of model accuracy (```F1-score```), model training time, and model inference time. When setting the weights of these three importance factors, you need to ensure that they are all greater than 0 and their sum is equal to 1.