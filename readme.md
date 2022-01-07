# how to use
## 1.in DataSetGen to generate Dataset
    (1) Dataset.csv is modified from Dataset_origin.csv which is make
    by Johns Hopkins University in 
    https://github.com/CSSEGISandData/COVID-19  
    (2) dataset_64_create.py is to create a input 64 output 16 dataset
    (3) DataNormalize.py is to Normalize the Dataset
    (4) this two datasets are over 100MB, and place in directory(folder)
    upper the Content root.

## 2.in input64_16,input671_30 linear and conv model are train
    (1) train_linear.py, train_conv.py are to train model with normalized 
    Dataset.
    (2) model_val.py,model_val_nor.py is to see model perform.
    (3) logs are over 100MB and place in directory(folder)
    upper the Content root.
## 2021.12.26 64input_16output
    First try to input all 702 days to train and test
    use 702-30 days as input and output 30 days to predict
    However, it might be not practical to use so many days as input
    because the days to early are to far away to influence the input
    as a consequence, it might be better to use 64 days or 32 days
    as input.
