# OM NAMO GANAPATHAYEN NAMAHA
# Copyright The IBM Tuning Team
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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from collections import deque
from typing import Any

# Third Party
from transformers import TrainerState
from transformers.utils import logging
from string import punctuation
import re
import torch 

# Local
from tuning.trainercontroller.controllermetrics.metricshandler import MetricHandler

METRICS_KEY = "metrics"
LOG_LOSS_KEY = "loss"
TRAINING_LOSS_KEY = "training_loss"
WINDOW_SIZE = "window_size"
STEP_KEY = "steps"
BATCH_KEY = "batch"
EPOCH_KEY = "epoch"

logger = logging.get_logger(__name__)

class BatchProcess(MetricHandler):
    """Implements the controller metric which evaluates loss-per-step"""

    def __init__(self, window_size=1, **kwargs):
        """Initializes the metric handler, by registering the event \
            list and arguments with base handler.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        super().__init__(events=["on_step_end_with_batch_data"], **kwargs)

    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are \
            compatible with the computation of this metric.

        Returns:
            bool
        """
        return True

    def compute(self, state: TrainerState = None, **kwargs) -> Any:
        """Exposes  the window of loss and metrics values in the log.

        Args:
            state: TrainerState object
            kwargs: Remaining event arguments

        Returns:
            Any. The exposed variables are returned here.
        """
        SUMMARY_BEGIN = 110
        SUMMARY_LIMIT = SUMMARY_BEGIN+100
        data = {}
        batch_data = kwargs["batch_data"]
        input_ids = batch_data['input_ids']
        tokenizer = kwargs["tokenizer"]
        decoded_batch = []
        for input_id in input_ids:
            decoded_data = tokenizer.decode(input_id)
            summary = decoded_data
            decoded_batch.append(summary)
        data = {
                    BATCH_KEY: decoded_batch,
                    "batch_data": batch_data
                }
        return data
