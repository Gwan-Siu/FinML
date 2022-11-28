FinML

### Install dependencies

1. This project is dependent on the [Qlib](https://github.com/microsoft/qlib), so you need to install the Qlib first. Please follow the steps to install from source.

- 1.1 Install the dependencies

 ```bash
  pip install numpy
  pip install --upgrade  cython
 ```

- 1.2 Clone the repository and install ``Qlib`` as follows.

```bash
  git clone https://github.com/microsoft/qlib.git && cd qlib
  pip install .
```

2. Install pytorch >= 1.7.0. You can refer to the [link](https://pytorch.org/get-started/locally/) and follow the instruction to install pytorch.

3. Install [TorchMertrics](https://torchmetrics.readthedocs.io/en/stable/).


### Dataset Preparation

1. We use the data_collector.py in Qlib to download the stock data from YahooFinance. 

```
cd qlib/scripts/data_collector/yahoo/
```

2. Download data to csv: `python scripts/data_collector/yahoo/collector.py download_data`.

- examples:

```bash
# cn 1d data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region CN
# cn 1min data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data_1min --delay 1 --interval 1min --region CN

# us 1d data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region US
# us 1min data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data_1min --delay 1 --interval 1min --region US

# in 1d data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/in_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region IN
# in 1min data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/in_data_1min --delay 1 --interval 1min --region IN

# br 1d data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/br_data --start 2003-01-03 --end 2022-03-01 --delay 1 --interval 1d --region BR
# br 1min data
python collector.py download_data --source_dir ~/.qlib/stock_data/source/br_data_1min --delay 1 --interval 1min --region BR
```

3. Normalize data: `python scripts/data_collector/yahoo/collector.py normalize_data`

- examples:

```bash
# normalize 1d cn
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_data --normalize_dir ~/.qlib/stock_data/source/cn_1d_nor --region CN --interval 1d

# normalize 1min cn
python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/cn_data_1min --normalize_dir ~/.qlib/stock_data/source/cn_1min_nor --region CN --interval 1min

# normalize 1d br
python scripts/data_collector/yahoo/collector.py normalize_data --source_dir ~/.qlib/stock_data/source/br_data --normalize_dir ~/.qlib/stock_data/source/br_1d_nor --region BR --interval 1d

# normalize 1min br
python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/br_data --source_dir ~/.qlib/stock_data/source/br_data_1min --normalize_dir ~/.qlib/stock_data/source/br_1min_nor --region BR --interval 1min
```


4. dump data: `python scripts/dump_bin.py dump_all`

- examples:

```bash
# dump 1d cn
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/cn_data --freq day --exclude_fields date,symbol
# dump 1min cn
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1min_nor --qlib_dir ~/.qlib/qlib_data/cn_data_1min --freq 1min --exclude_fields date,symbol
```

### Set up the deep learning models.

Please refer to `model/arch/lstm_arch.py`. The file name must be the format `xxxx_arch.py`. In this project, we use two pubilc datasets `Alpha158` and 'Alpha360'. The files `xxx_ts_arch.py` are used for the 'Alpha158' datset, while the files `xxxx_arch.py` are used for the `Alpha360` dataset.

- `Alpha158` dataset is a tabular dataset. There are less spatial relationships between different features. Each feature are carefully designed by human (a.k.a feature engineering).
- `Alpha360` dataset contains raw price and volue data without much feature engineering. There are strong strong spatial relationships between the features in the time dimension. 

### Set up the .yml file

Please refer to the files in the folder 'options'.


### Run the file `main.py`

```
python main.py
```

### Experiment Results

|  | LSTM | ALSTM | GATS | Transformer | Localformer |
| :-----:| :----: | :----: | :----: | :----: | :----: |
| IC | 0.0410 | 0.0404 | 0.0276 | 0.0257 | 0.0213 |
| Rank IC | 0.0359 | 0.0376 | 0.0292 | 0.0211 | 0.0247 |


- IC: information coefficient
- Rank IC: rank information coefficient


