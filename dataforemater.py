# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
import numpy as np
import pandas as pd
import sklearn.preprocessing


class DataAttributes:

    @property
    def _data_def(self):

        return {"traffic": {'id': 'id', 'target':
                            'values', 'covariates': [],},
                "electricity": {'id': 'id', 'target': 'power_usage', 'covariates': []},
                "solar": {'id': 'id', 'target': 'Power(MW)',
                 'covariates': [], 'time': 'hours_from_start'},
                "air_quality": {'id': 'id', 'target': 'NO2',
                       'covariates': ['CO', 'TEMP']},
                "watershed": {'id': 'id', 'target': 'Conductivity',
                     'covariates': ['Q']},
                "exchange": {'id': 'id', 'target': 'OT',
                    'covariates': ['0', '1', '2', '3', '4', '5']}}

    def __init__(self, exp_name):

        col_def = self._data_def[exp_name]
        self.id = col_def['id']
        self.target = col_def['target']
        self.covariates = col_def['covariates']


class DataFormatter:
    """Defines and formats data_set for the electricity dataset.
    Note that per-entity z-score normalization is used here, and is implemented
    across functions.
    Attributes:
    column_definition: Defines input and data_set type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    def __init__(self, exp_name: str):
        """Initialises formatter."""

        self.identifiers = None
        self.real_scalers = None
        self.target_scaler = None

        self.column_definition = DataAttributes(exp_name)
        self.id_column = self.column_definition.id
        self.target_column = self.column_definition.target

        self.real_inputs = []
        self.real_inputs.append(self.target_column)
        for covar in self.column_definition.covariates:
            self.real_inputs.append(covar)

        print(self.real_inputs)

    def transform_data(self, df):

        print('Formatting data.')

        return self.set_scalers(df)

    def set_scalers(self, df):

        """Calibrates scalers using the data_set supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with data_set...')

        self.real_scalers = {}
        self.target_scaler = {}

        real_data = df[self.real_inputs].values
        targets = df[self.target_column].values
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        if len(real_data.shape) == 1:
            real_data = real_data.reshape(-1, 1)
        self.real_scalers[self.id_column] = sklearn.preprocessing.StandardScaler().fit(real_data)
        self.target_scaler[self.id_column] = sklearn.preprocessing.StandardScaler().fit(targets)
        data = df.copy()
        data[self.real_inputs] = self.real_scalers[self.id_column].transform(real_data)

        return data

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """

        if self.target_scaler is None:
            raise ValueError('Scalers have not been set!')

        output = self.target_scaler[[self.id_column]].inverse_transform(predictions)

        return output