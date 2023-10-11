# FusionFL (FusionFL: A statement-level feature fusion based fault localization approach)

## I. Requirements For FusionFL

* [Python 3.7.6](https://www.python.org/downloads/)
* [PyTorch-1.12.1](https://pytorch.org/)

## II. Download Dataset

1. **Click the following url link and download the necessary dataset used in this research.**
   [cooked.tar.gz](https://mega.nz/file/dzVBRT4A#NnN7TAysQEwdLuBmpvstwsAXOBSSacXJoOWvKsFXePM)
   [raw_data.tar.gz](https://mega.nz/file/NjlESDBI#1QNHoto4djBzhPujsLr-h26210Q_SMMkyiqS360bnrk)
   [model_save.tar.gz](https://mega.nz/file/EjUSDYKY#GO9Kt0iSFjh3kM48AVbaArAvbhFsEiAUXLumG_M2Y2A)
2. **Unzip each file and put the extracted data in the corresponding path: **
   * **upzip** `cooked.tar.gz` into `./data/`
   * **upzip** `raw_data.tar.gz` into `./data/`
   * **upzip** `model_save.tar.gz` into `./code/`

## III. Perform Fault Localization Result with Already Trained Model

1. **Enter into **`code` folder

```
  cd code
```

2. **use **`test_group.py` to evaluate trained model

```
  python test_group.py
```

3. **See the repair results in **`./code/result/`

*notice that the code only support for gpu mode*

## IV. Train New Models for FL

1. **Enter into **`data` folder

```
  cd data
```

2. **Run **`preprocess.py`, you can decide the line_len (**l_s**) and word_len (**l_t**)

```
  python preprocess.py --data squeeze_5e-3 --line-len 40 --word-len 20
```

3. **Goto **`code`  folder and run `pipeline_group.py` for 10-cross validation data

```
  cd ../code
  python pipeline_group.py squeeze_40_20_new
```

4. **Run **`run_group.py` to train and test

```
  python run_group.py
```

* **For more training args option, please see **`run_group.py`
