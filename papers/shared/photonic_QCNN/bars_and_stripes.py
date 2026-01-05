# Copyright 2024 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def generate_bars_and_stripes(n_samples, height, width, noise_std):
    """Data generation for Bars and Stripes with optional Gaussian noise."""
    x = np.ones([n_samples, 1, height, width]) * -1
    y = np.zeros([n_samples])

    for i in range(len(x)):
        if np.random.rand() > 0.5:
            rows = np.where(np.random.rand(height) > 0.5)[0]
            x[i, 0, rows, :] = 1.0
            y[i] = +1
        else:
            columns = np.where(np.random.rand(width) > 0.5)[0]
            x[i, 0, :, columns] = 1.0
            y[i] = -1
        x[i, 0] = x[i, 0] + np.random.normal(0, noise_std, size=x[i, 0].shape)

    return x, y
