# Adapted weighted aggregation in Federated Learning

PyTorch implementation of "Adapted weighted aggregation in Federated Learning"


## Make dataset
```
python make_dataset.py with server_user make_target=colored_mnist
```

## Training Vanilla in Federated Setting
```
python train_fl_LfF.py with server_user colored_mnist skewed3 severity4
```

## Training LfF in Federated Setting
```
python train_fl_LfF.py with server_user colored_mnist skewed3 severity4
```

## Training BiasAdv in Federated Setting
```
python fedrated_train.py with server_user colored_mnist skewed3 severity4
```

## Training DFA in Federated Setting
[Learning-Debiased-Disentangled-FL
](https://github.com/Yitong999/Learning-Debiased-Disentangled-FL)

