import numpy as np

from data.dataset import TSDataset



class DriftDataset(TSDataset):
    """Add a constant offset to a random feature of the data."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.offset = self.set_params(severity)

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(int(self.n_features * prct_affected_sensors), len(self.continuous_features))
        affected_sensors = self.rng.choice(self.continuous_features, n_affected_sensors, replace=False)  # continuous features only

        min_offset = 0.
        max_offset = 1.
        offset = min_offset + severity * (max_offset - min_offset)
        # offset = np.linspace(offset, 0, self.input_len)

        return affected_sensors, offset

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[:, sensor_idx] += self.offset
            # y[:, sensor_idx] += self.offset
        return x, y


class DyingSignalDataset(TSDataset):
    """Multiply the a random feature of the data with a constant factor."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        # self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)  # no disturbance if flat sensor is hit, therefore we increase the percentage here
        self.affected_sensors, self.factor = self.set_params(severity)

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(int(self.n_features * prct_affected_sensors), len(self.continuous_features))
        affected_sensors = self.rng.choice(self.continuous_features, n_affected_sensors, replace=False)  # continuous features only

        min_factor = 1.
        max_factor = 0.
        factor = min_factor + severity * (max_factor - min_factor)
        # factor = np.linspace(factor, 1, self.input_len)

        return affected_sensors, factor

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[:, sensor_idx] *= self.factor
            # y[:, sensor_idx] *= self.factor
        return x, y


class NoiseDataset(TSDataset):
    """Add Gaussian noise to the data."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.sd = self.set_params(severity)
        self.noise = self._create_noise()

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(int(self.n_features * prct_affected_sensors), len(self.continuous_features))
        affected_sensors = self.rng.choice(self.continuous_features, n_affected_sensors, replace=False)  # continuous features only

        min_sd = 0.
        max_sd = 1.
        sd = min_sd + severity * (max_sd - min_sd)

        return affected_sensors, sd

    def _create_noise(self):
        full_noise = self.rng.normal(0, self.sd, (self.n_samples, self.input_len, self.n_features))
        noise = np.zeros((self.n_samples, self.input_len, self.n_features))
        for i in self.affected_sensors:
            idx = self.df.columns.get_loc(i)
            noise[:, :, idx] = full_noise[:, :, idx]
        return noise.astype(np.float32)
        
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x_with_noise = x + self.noise[index]
        return x_with_noise, y


class FlatSensorDataset(TSDataset):
    """Set a random sensor to the last value for a random duration."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05,  **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)  # no disturbance if flat sensor is hit, therefore we increase the percentage here
        self.affected_sensors, self.flat_duration = self.set_params(severity)
        self.flat_start_pos = self.rng.integers(1, self.input_len - self.flat_duration + 2, size=(self.n_samples, self.n_features))  # only affected sensors sample from this

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = int(self.n_features * prct_affected_sensors)
        affected_sensors = self.rng.choice(self.feature_names, n_affected_sensors, replace=False)

        min_flat_duration = 1
        max_flat_duration = self.input_len
        flat_duration = int(min_flat_duration + severity * (max_flat_duration - min_flat_duration))

        return affected_sensors, flat_duration

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            start_pos = self.flat_start_pos[index, sensor_idx]
            end_pos = start_pos + self.flat_duration
            last_valid_value = x[start_pos - 1, sensor_idx]
            x[start_pos:end_pos, sensor_idx] = last_valid_value
        return x, y


class MissingDataDataset(TSDataset):
    """Remove a random time window from the data."""
    def __init__(self, severity=1., target_prct_affected_sensors=1., **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.affected_sensors, self.missing_duration = self.set_params(severity)
        self.missing_start_pos = self.rng.integers(0, self.input_len - self.missing_duration, size=(self.n_samples))

    def set_params(self, severity):
        affected_sensors = self.feature_names.values  # all sensors are affected, regardless of choice of targetr_prct_affected_sensors

        min_missing_duration = 1
        max_missing_duration = int(self.input_len * 0.5)
        missing_duration = min_missing_duration + int(severity * (max_missing_duration - min_missing_duration))

        return affected_sensors, missing_duration

    def __getitem__(self, index):
        start_idx = self.sample_idxs[index]
        end_idx = start_idx + self.input_len + self.target_len + self.missing_duration
        df_ = self.df.iloc[start_idx:end_idx]

        missing_start = self.missing_start_pos[index]
        missing_end = missing_start + self.missing_duration
        df_ = df_.drop(df_.index[missing_start:missing_end])

        x = df_.iloc[:self.input_len].to_numpy().astype(np.float32)
        y = df_.iloc[self.input_len:].to_numpy().astype(np.float32)

        return x, y


class OutlierDataset(TSDataset):
    """Add an outlier to a random sensor of the data."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.hickup_value = self.set_params(severity)
        self.fault_mask = self._create_fault_mask()

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(int(self.n_features * prct_affected_sensors), len(self.continuous_features))
        affected_sensors = self.rng.choice(self.continuous_features, n_affected_sensors, replace=False)  # continuous features only

        min_hickup_value = 1
        max_hickup_value = 25
        hickup_value = min_hickup_value + severity * (max_hickup_value - min_hickup_value)

        return affected_sensors, hickup_value

    def _create_fault_mask(self):
        fault_mask = np.zeros((self.n_samples, self.input_len, self.n_features), dtype=np.float32)
        for sample in range(self.n_samples):
            for sensor in self.affected_sensors:
                sensor_idx = self.df.columns.get_loc(sensor)
                hickup_postion = self.rng.integers(1, self.input_len)
                fault_mask[sample, hickup_postion, sensor_idx] = self.hickup_value
        return fault_mask

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x_with_fault = x + self.fault_mask[index]
        return x_with_fault, y


class FasterSamplingDataset(TSDataset):
    """Irregularly sample the data by warping the time axis of the input sequence.
    The time axis is warped by a factor between 1 and 3 during a fixed duration.
    After the warped time frame, the sensor remains flat until synchronizing with the original time axis.
    """
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)  # no disturbance if flat sensor is hit, therefore we increase the percentage here
        self.affected_sensors, self.warp_factor, self.warp_duration = self.set_params(severity)
        self.warp_start_pos = self.rng.integers(0, int(self.input_len * 0.5), size=(self.n_samples))  # only affected sensors sample from this. 
        # only first half of the input_len is warped to avoid leakage. same start point for all affected sensors of a sample

    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = int(self.n_features * prct_affected_sensors)
        affected_sensors = self.rng.choice(self.feature_names, n_affected_sensors, replace=False)

        min_warp_factor = 1.
        max_warp_factor = 5.
        warp_factor = min_warp_factor + severity * (max_warp_factor - min_warp_factor)

        warp_duration = int(self.input_len * 0.5)

        return affected_sensors, warp_factor, warp_duration

    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        original_time_index = np.arange(self.input_len)
        irreg_time = np.full(self.warp_duration, self.warp_factor)
        irreg_time_index = np.cumsum(irreg_time) + self.warp_start_pos[index] - 1

        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[self.warp_start_pos[index]:(self.warp_start_pos[index] + self.warp_duration), sensor_idx] = np.interp(irreg_time_index, original_time_index, x[:, sensor_idx])
        
        return x, y


class SlowerSamplingDataset(TSDataset):
    """Irregularly sample the data by warping the time axis of the input sequence.
    The time axis is warped by a factor between 1 and 0 during a fixed duration.
    After the warped time frame, the sensor immediately jumps back to the original time axis.
    """
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = min(target_prct_affected_sensors * 5, 1)  # no disturbance if flat sensor is hit, therefore we increase the percentage here
        self.affected_sensors, self.warp_factor, self.warp_duration = self.set_params(severity)
        self.warp_start_pos = self.rng.integers(0, int(self.input_len * 0.5), size=(self.n_samples))  # only affected sensors sample from this. 
        # only first half of the input_len is warped to avoid leakage. same start point for all affected sensors of a sample
        
    def set_params(self, severity):
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = int(self.n_features * prct_affected_sensors)
        affected_sensors = self.rng.choice(self.feature_names, n_affected_sensors, replace=False)

        min_warp_factor = 1.
        max_warp_factor = 0.
        warp_factor = min_warp_factor + severity * (max_warp_factor - min_warp_factor)

        warp_duration = int(self.input_len * 0.5)

        return affected_sensors, warp_factor, warp_duration

    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        original_time_index = np.arange(self.input_len)
        irreg_time = np.full(self.warp_duration, self.warp_factor)
        irreg_time_index = np.cumsum(irreg_time) + self.warp_start_pos[index] - 1

        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            x[self.warp_start_pos[index]:(self.warp_start_pos[index] + self.warp_duration), sensor_idx] = np.interp(irreg_time_index, original_time_index, x[:, sensor_idx])
        
        return x, y
    

class WrongDiscreteValueDataset(TSDataset):
    """A discrete sensor or actuator shows a wrong value."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.wrong_duration = self.set_params(severity)
        self.wrong_start_pos = self.rng.integers(1, self.input_len - self.wrong_duration + 2, size=(self.n_samples, self.n_features))  # only affected sensors sample from this

    def set_params(self, severity):
        if len(self.discrete_features) == 0:
            raise ValueError("No discrete features available.")
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(int(self.n_features * prct_affected_sensors), len(self.discrete_features))
        affected_sensors = self.rng.choice(self.discrete_features, n_affected_sensors, replace=False)  # discrete features only

        min_wrong_duration = 1
        max_wrong_duration = int(self.input_len / 10)
        wrong_duration = int(min_wrong_duration + severity * (max_wrong_duration - min_wrong_duration))

        return affected_sensors, wrong_duration

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            start_pos = self.wrong_start_pos[index, sensor_idx]
            end_pos = start_pos + self.wrong_duration
            x[start_pos:end_pos, sensor_idx] = 2  # set a fixed wrong value
        return x, y


class OscillatingSensorDataset(TSDataset):
    """A discrete sensor or actuator oscillates between two values."""
    def __init__(self, severity=1., target_prct_affected_sensors=0.05, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= severity <= 1, "Severity must be between 0 and 1."
        self.severity = severity
        self.target_prct_affected_sensors = target_prct_affected_sensors
        self.affected_sensors, self.osc_duration = self.set_params(severity)
        self.osc_start_pos = self.rng.integers(1, self.input_len - self.osc_duration + 1, size=(self.n_samples, self.n_features))  # only affected sensors sample from this

    def set_params(self, severity):
        if len(self.discrete_features) == 0:
            raise ValueError("No discrete features available.")
        min_prct_affected_sensors = 1 / self.n_features + 1e-9  # at least one sensor must be affected
        prct_affected_sensors = max(min_prct_affected_sensors, self.target_prct_affected_sensors)
        n_affected_sensors = min(int(self.n_features * prct_affected_sensors), len(self.discrete_features))
        affected_sensors = self.rng.choice(self.discrete_features, n_affected_sensors, replace=False)  # discrete features only

        min_osc_duration = 1
        max_osc_duration = int((self.input_len - 1) / 10)  # start value is not affected (technically it is, but it is not visible)
        osc_duration = int(min_osc_duration + severity * (max_osc_duration - min_osc_duration))

        return affected_sensors, osc_duration

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        for sensor in self.affected_sensors:
            sensor_idx = self.df.columns.get_loc(sensor)
            start_pos = self.osc_start_pos[index, sensor_idx]
            end_pos = start_pos + self.osc_duration

            last_value = x[start_pos - 1, sensor_idx]
            if self.df[sensor].nunique() == 1:
                wrong_value = 1  # if there was only one value, it would be standardized to 0, so set this to 1. Why is this sensor in the data to begin with?
            else:
                unique_values = self.df[sensor].unique().astype(np.float32)
                filtered_values = unique_values[unique_values != last_value]  # force a different value
                wrong_value = self.rng.choice(filtered_values)
            oscillating_values = self.rng.choice([last_value, wrong_value], size=self.osc_duration, replace=True)
            x[start_pos:end_pos, sensor_idx] = oscillating_values

        return x, y
