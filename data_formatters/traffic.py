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

import sklearn.preprocessing


class TrafficFormatter:

    @property
    def _column_definition(self):
        """Defines order, input type and data_set type of each column."""
        return {'id': 'id', 'target': 'power_usage', 'covariates': []}

    def __init__(self):

        self.identifiers = None
        self.real_scalers = None
        self.target_scaler = None

        self.id_column = self._column_definition['id']
        self.target_column = self._column_definition['target']

        self.real_inputs = []
        self.real_inputs.append(self.target_column)
        for covar in self._column_definition["covariates"]:
            self.real_inputs.append(covar)

    def transform_data(self, df):
        """Splits data_set frame into training-validation-test data_set frames.
        This also calibrates scaling object, and transforms data_set for each split.
        Args:
          df: Source data_set frame to split.
          valid_boundary: Starting year for validation data_set
          test_boundary: Starting year for test data_set
        Returns:
          Tuple of transformed (train, valid, test) data_set.
        """

        print('Formatting train-valid-test splits.')

        self.set_scalers(df)

        return self.transform_inputs(df)

    def set_scalers(self, df):
        """Calibrates scalers using the data_set supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data_set...')

        self.identifiers = list(df[self.id_column].unique())

        data = df[self.real_inputs].values
        self.real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self.target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[self.target_column]].values)  # used for predictions

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data_set frame.
        """
        output = df.copy()

        if self.real_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Format real inputs
        output[self.real_inputs] = self.real_scalers.transform(df[self.real_inputs].values)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """

        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'identifier'}:
                output[col] = self.target_scaler.inverse_transform(predictions[col])

        return output
