import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import TSDataset
from data.disturbances import *



class TSDataModule(pl.LightningDataModule):
    """Data module for time series data
    Args:
        file_path (str): path to csv file containing the data
        input_len (int): length of input sequence
        target_len (int): length of target sequence
        stride (int): stride between samples. Only used if n_samples is None and samples are thus drawn sequentially.
        n_train_samples (int): number of train samples to draw. If None, all possible samples are drawn sequentially. Else, samples are drawn randomly.
        n_val_samples (int): number of val/ samples to draw. If None, all possible samples are drawn sequentially. Else, samples are drawn randomly.
        n_test_samples (int): number of test samples to draw. If None, all possible samples are drawn sequentially. Else, samples are drawn randomly.
        n_severity_levels (int): number of severity levels for each test scenario
        prct_affected_sensors (float): target percentage of sensors affected by perturbations
        train_split (float): fraction of training data
        val_split (float): fraction of validation data
        purged_fraction (float): fraction of all data to be purged of the start of the validation and test sets. This is done to avoid leakage.
        batch_size (int): batch size
        num_workers (int): number of workers for dataloader
        pin_memory (bool): pin memory for dataloader
        persistent_workers (bool): persistent workers for dataloader
        seed (int): seed for reproducibility
        s3_endpoint (str): s3 storage url
    """
    def __init__(self,
                 file_path,
                 input_len=100, target_len=20, stride=1,
                 n_train_samples=None, n_val_samples=None, n_test_samples=None, n_severity_levels=101, prct_affected_sensors=0.05,
                 train_split=0.7, val_split=0.15, purged_fraction=0.01,
                 batch_size=64, num_workers=None, pin_memory=True, persistent_workers=False, seed=None,
                 s3_endpoint='http://minio.minio'):
        super().__init__()

        self.storage_options = {
            "key": os.environ.get('AWS_ACCESS_KEY_ID'),
            "secret": os.environ.get('AWS_SECRET_ACCESS_KEY'),
            "client_kwargs": {"endpoint_url": s3_endpoint}}
        self.file_path = file_path

        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride

        self.n_features = self._get_n_features()
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.n_severity_levels = n_severity_levels
        self.prct_affected_sensors = prct_affected_sensors

        self.train_split = train_split
        self.val_split = val_split
        self.purged_fraction = purged_fraction

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # datasets, populated in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_test_dict = {}  # each scenario will have a list of datasets, one for each severity level
        
        # dictionary with the test scenarios and their corresponding dataset classes
        self.dataset_mapping = {
            'normal': TSDataset,  # 'normal' is the default scenario, i.e. no perturbation is applied
            'drift': DriftDataset,
            'dying_signal': DyingSignalDataset,
            'noise': NoiseDataset,
            'flat_sensor': FlatSensorDataset,
            'missing_data': MissingDataDataset,
            'faster_sampling': FasterSamplingDataset,
            'slower_sampling': SlowerSamplingDataset,
            'outlier': OutlierDataset,
            'wrong_discrete_value': WrongDiscreteValueDataset,  # removed if there are no discrete features in the dataset
            'oscillating_sensor': OscillatingSensorDataset,  # removeed used if there are no discrete features in the dataset
        }
        self.test_dataloader_mapping = pd.DataFrame(columns=["dataloader_idx", "scenario", "severity"]).set_index("dataloader_idx")
        self.test_scenarios = self.dataset_mapping.keys()  # scenarios to be used for validation and testing

        # mean and standard deviation values of training data for scaling, they are set in setup()
        self.train_mean_vals = None  
        self.train_sd_vals = None

        self.seed = seed

    def _get_n_features(self):
        # unfortunately we must read some lines of the file to get the number of features before setup() is called
        if self.file_path.endswith(".parquet"):
            # it is even worse here, we are reading the whole parquet file
            # since we we too lacy to make reading a couple of line work with
            # remote s3 storage
            if self.file_path.startswith('s3://'):
                df = pd.read_parquet(self.file_path, storage_options=self.storage_options)
            else:
                df = pd.read_parquet(self.file_path)
        elif self.file_path.endswith(".csv"):
            if self.file_path.startswith('s3://'):
                df = pd.read_csv(self.file_path, nrows=3, storage_options=self.storage_options)
            else:
                df = pd.read_csv(self.file_path, nrows=3)
            try:
                df.set_index(pd.to_datetime(df.iloc[:,0], format="%Y-%m-%d %H:%M:%S"), inplace=True)
                df.drop(df.columns[0], axis=1, inplace=True)
            except (ValueError, TypeError):
                pass
        else:
            raise ValueError("File format not supported.")
        n_features = df.shape[1]
        del df
        return n_features


    def _get_split_indices(self, len_df):
        train_end_idx = int(len_df * self.train_split)
        val_start_idx = train_end_idx
        val_end_idx = int(len_df * (self.train_split + self.val_split))

        purged_val_start_idx = val_start_idx + int(len_df * self.purged_fraction)
        purged_test_start_idx = val_end_idx + int(len_df * self.purged_fraction)

        # ensure indices are within the bounds of the dataframe
        assert train_end_idx < len_df
        assert purged_val_start_idx < len_df
        assert val_end_idx < len_df
        assert purged_test_start_idx < len_df

        return train_end_idx, purged_val_start_idx, val_end_idx, purged_test_start_idx

    def setup(self, stage=None) -> None:
        # read the complete dataset
        verbose = self.trainer.testing and self.trainer.test_loop.verbose
        if verbose: print("Setting up the dataset.")
        if self.file_path.endswith(".parquet"):
            if self.file_path.startswith('s3://'):
                df_all = pd.read_parquet(self.file_path, storage_options=self.storage_options)
            else:
                df_all = pd.read_parquet(self.file_path)
        elif self.file_path.endswith(".csv"):
            if self.file_path.startswith('s3://'):
                df_all = pd.read_csv(self.file_path, storage_options=self.storage_options)
            else:
                df_all = pd.read_csv(self.file_path)
            try:
                df_all.set_index(pd.to_datetime(df_all.iloc[:,0], format="%Y-%m-%d %H:%M:%S"), inplace=True)
                df_all.drop(df_all.columns[0], axis=1, inplace=True)
            except (ValueError, TypeError):
                print("First column is not a valid datetime. Keeping the default index.")
        else:
            raise ValueError("File format not supported.")
        # fcols = df_all.select_dtypes("float").columns
        # df_all[fcols] = df_all[fcols].apply(pd.to_numeric, downcast="float")
        # icols = df_all.select_dtypes("integer").columns
        # df_all[icols] = df_all[icols].apply(pd.to_numeric, downcast="integer")
            
        self.n_features = df_all.shape[1]

        # split the dataset into train, val and test
        train_end, purged_val_start, val_end, purged_test_start = self._get_split_indices(len(df_all))
        df_train = df_all.iloc[:train_end, :]
        df_val = df_all.iloc[purged_val_start:val_end, :]
        df_test = df_all.iloc[purged_test_start:, :]  if self.n_test_samples is not None else None
        del df_all

        # standardize the datasets using the values of the training data
        self.train_mean_vals = df_train.mean()
        self.train_sd_vals = df_train.std()
        self.train_sd_vals.replace(0, 1.0, inplace=True)
        df_train = (df_train - self.train_mean_vals) / self.train_sd_vals
        df_val = (df_val - self.train_mean_vals) / self.train_sd_vals
        df_test = (df_test - self.train_mean_vals) / self.train_sd_vals if df_test is not None else None

        # create the datasets
        self.ds_train = TSDataset(
            df=df_train,
            input_len=self.input_len,
            target_len=self.target_len,
            stride=self.stride,
            n_samples=self.n_train_samples,
            seed=self.seed
        )
        self.continuous_features = self.ds_train.continuous_features
        self.discrete_features = self.ds_train.discrete_features
        del df_train

        self.ds_val = TSDataset(
            df=df_val,
            input_len=self.input_len,
            target_len=self.target_len,
            stride=self.stride,
            n_samples=self.n_val_samples,
            continuous_features=self.continuous_features,
            discrete_features=self.discrete_features,
            seed=self.seed
        )
        del df_val

        # create datasets for each test scenario
        if self.trainer.testing and self.n_test_samples is not None:
            if len(self.ds_train.discrete_features) == 0:
                self.dataset_mapping.pop('wrong_discrete_value', None)
                self.dataset_mapping.pop('oscillating_sensor', None)
                if verbose: print("Removing test scenarios 'wrong_value' and 'oscillation' because there are no discrete features in the dataset.")
            dataloader_idx = 0
            for i, scenario in enumerate(self.dataset_mapping.keys()):
                if verbose: print(f"Setting up test dataloaders for the test scenario '{scenario}' ({i+1}/{len(self.dataset_mapping)}).")
                if scenario == "normal":
                    self.test_dataloader_mapping.loc[dataloader_idx] = [scenario, np.nan]
                    dataloader_idx += 1
                    self.ds_test_dict[scenario] = [self.dataset_mapping[scenario](
                            df=df_test,
                            input_len=self.input_len,
                            target_len=self.target_len,
                            stride=self.stride,
                            n_samples=self.n_test_samples,
                            continuous_features=self.continuous_features,
                            discrete_features=self.discrete_features,
                            seed=self.seed
                        )]
                else:
                    scenario_dataloaders = []
                    for severity in np.linspace(0, 1, self.n_severity_levels):
                        self.test_dataloader_mapping.loc[dataloader_idx] = [scenario, severity]
                        dataloader_idx += 1
                        scenario_dataloaders.append(
                            self.dataset_mapping[scenario](
                                df=df_test,
                                severity=severity,
                                target_prct_affected_sensors=self.prct_affected_sensors,
                                input_len=self.input_len,
                                target_len=self.target_len,
                                stride=self.stride,
                                n_samples=self.n_test_samples,
                                continuous_features=self.continuous_features,
                                discrete_features=self.discrete_features,
                                seed=self.seed
                            )
                        )
                    self.ds_test_dict[scenario] = scenario_dataloaders
            del df_test
        if verbose: print("Done.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        test_dl_list = [
            DataLoader(
                self.ds_test_dict[scenario][severity_level],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=False
                )
        for scenario in self.test_scenarios
        for severity_level in range(len(self.ds_test_dict[scenario]))
        ]
        return test_dl_list
