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

from forecastblurdenoise.Utils.base import DataTypes, InputTypes
from data_formatters.electricity import ElectricityFormatter

DataFormatter = ElectricityFormatter


class ExchangeFormatter(DataFormatter):
    @property
    def _column_definition(self):
        return {'id': 'id', 'target': 'OT', 'covariates': ['0', '1', '2', '3', '4', '5']}
