import numpy as np
import pandas as pd
from torch.utils.data import Dataset



class TSDataset(Dataset):
    """Time Series Dataset
    A sample consists of a (random) time window + consecutive time horizon.
    Args:
        df (pd.DataFrame): dataframe containing the data
        file_path (str): path to csv file containing the data
        input_len (int): length of input sequence
        target_len (int): length of target sequence
        stride (int): stride between samples. Only used if n_samples is None and samples are thus drawn sequentially.
        n_samples (int): number of samples to draw. If None, all possible samples are drawn sequentially. Else, samples are drawn randomly.
        mean_vals (np.array): mean values for each feature. If given, data is rescaled to have zero mean.
        sd_vals (np.array): standard deviation values for each feature. If given, data is rescaled to have unit variance.
        continuous_features (list): list of continuous features
        discrete_features (list): list of discrete features
        seed (int): seed for reproducibility
    """
    def __init__(self,
                 df=None,
                 file_path=None,
                 input_len=90,
                 target_len=30,
                 stride=1,
                 n_samples=None,
                 mean_vals=None,
                 sd_vals=None,
                 continuous_features=None,
                 discrete_features=None,
                 seed=42
                 ):
        super().__init__()
        if df is not None:
            self.df = df
        elif file_path is not None:
            if file_path.endswith(".parquet"):
                self.df = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError("File format not supported.")
            try:  # Try to convert the first column to datetime. If not possible, ignore it.
                self.df.set_index(pd.to_datetime(self.df.iloc[:,0], format="%Y-%m-%d %H:%M:%S"), inplace=True)
                self.df.drop(self.df.columns[0], axis=1, inplace=True)
            except ValueError:
                pass
        else:
            raise ValueError("Either df or file_path must be given.")

        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride  # only used if n_samples is None and samples are drawn sequentially

        self.random_sampling = n_samples is not None
        self.n_samples = self.__len__() if n_samples is None else n_samples
        assert self.n_samples <= self.__len__(), "n_samples must be smaller than the number of possible samples."
        self.n_features = self.df.shape[1]
        self.feature_names = self.df.columns
        self.continuous_features, self.discrete_features = self.split_hybrid_data(continuous_features, discrete_features)

        # Rescale data if min and max values are given
        self.mean_vals = mean_vals
        self.sd_vals = sd_vals
        if mean_vals is not None and sd_vals is not None:
            self.scale_data()

        if seed is not None:
            self.rng = np.random.default_rng(seed)  # Using a local random number generator
        else:
            self.rng = np.random.default_rng()  # Default random generator without a fixed seed

        self.sample_idxs = self._create_sample_indices()

    def split_hybrid_data(self, continuous_features=None, discrete_features=None):
        """Split the time series data features into continuous and discrete features."""
        continuous_threshold = 32
        continuous_features = [feature for feature in self.df.columns if self.df[feature].nunique() > continuous_threshold] if continuous_features is None else continuous_features
        discrete_features = [feature for feature in self.df.columns if self.df[feature].nunique() <= continuous_threshold] if discrete_features is None else discrete_features
        assert len(continuous_features) + len(discrete_features) == self.n_features, "All features must be assigned to either continuous or discrete features."
        return continuous_features, discrete_features

    def set_scaler_params(self, mean_vals=None, sd_vals=None):
        """Set the parameters for scaling the data.
        Args:
            mean_vals (np.array): mean values for each feature. If given, data is rescaled to have zero mean.
            sd_vals (np.array): standard deviation values for each feature. If given, data is rescaled to have unit variance.
        """
        self.mean_vals = mean_vals if mean_vals is not None else self.df.mean()
        self.sd_vals = sd_vals if sd_vals is not None else self.df.std()

    def scale_data(self):
        """Scale data between min and max values."""
        assert self.mean_vals is not None and self.sd_vals is not None, "Mean and standard deviation values must be set first."
        # Avoid division by zero by replacing sd value of 0 with 1 (for constant features)
        self.sd_vals.replace(0, 1.0, inplace=True)
        # Standardize the data
        self.df = (self.df - self.mean_vals) / self.sd_vals

    def inverse_scale_data(self, scaled_data):
        df_ = pd.DataFrame(scaled_data, columns=self.df.columns)
        return (df_ * self.sd_vals) + self.mean_vals

    def _create_sample_indices(self):
        """Create an array of indices for sampling"""
        if self.random_sampling:
            sample_idxs = self.rng.integers(low=0, high=self.df.shape[0] - 2 * self.input_len - self.target_len, size=self.n_samples)  # -2*input_len because some perturbations might require more than input_len time steps
        else:
            max_n_samples = int((self.df.shape[0] - 2 * self.input_len - self.target_len) / self.stride) + 1  # -2*input_len because some perturbations might require more than input_len time steps
            sample_idxs = np.arange(max_n_samples) * self.stride
        return sample_idxs

    def __len__(self):
        """Number of samples"""
        if self.random_sampling:
            return self.n_samples
        else:
            return int((self.df.shape[0] - self.input_len - self.target_len) / self.stride) + 1

    def __getitem__(self, index):
        """Get one sample.
        A sample consists of a time window of length input_len and a consecutive time horizon of length target_len.
        Returns:
            x (np.array): input sequence
            y (np.array): target sequence
        """
        start_idx = self.sample_idxs[index]
        end_idx = start_idx + self.input_len + self.target_len
        df_ = self.df.iloc[start_idx:end_idx]
        x = df_.iloc[:self.input_len].to_numpy().astype(np.float32)
        y = df_.iloc[self.input_len:].to_numpy().astype(np.float32)
        del df_
        return x, y

