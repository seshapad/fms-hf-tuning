# Standard
import re

# Third Party
from transformers import TrainingArguments
from transformers.utils import logging

# Local
from .operation import Operation

logger = logging.get_logger(__name__)
logger.setLevel(level=logging.DEBUG)


class BatchInspect(Operation):
    """Operation that can be used to log useful information on specific events."""

    def __init__(self, log_level: str, spike_threshold: float, **kwargs):
        """Initializes the HuggingFace controls. In this init, the fields with `should_` of the
        transformers.TrainerControl data class are extracted, and for each of those fields, the
        control_action() method's pointer is set, and injected as a class member function.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        log_levels = logging.get_log_levels_dict()
        if log_level not in log_levels:
            raise ValueError(
                "Specified log_level [%s] is invalid for LogControl" % (log_level)
            )
        self.log_level = log_levels[log_level]
        # self.log_format = log_format
        self.spike_threshold = spike_threshold
        super().__init__(**kwargs)

    def _find_spikes(self, batch_data, decoded_batch, model, training_loss):
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        num_inputs = len(input_ids)
        for i in range(num_inputs):
            new_batch_data = {'input_ids': input_ids[i:i+1,], \
                            'attention_mask': attention_mask[i:i+1,], \
                            'labels': labels[i:i+1,]}
            # logger.warn(f"Input: {new_batch_data}")
            outputs = model(**new_batch_data)
            if outputs['loss'] > self.spike_threshold:
                logger.warn(f"Faulty sample input {i} -> SL: {outputs['loss']}, TL: {training_loss} -> \n\n{decoded_batch[i]}\n\n")

    def should_log(
        self,
        event_name: str = None,
        control_name: str = None,
        args: TrainingArguments = None,
        **kwargs,
    ):
        """This method peeks into the stack-frame of the caller to get the action the triggered
        a call to it. Using the name of the action, the value of the control is set.

        Args:
            control: TrainerControl. Data class for controls.
            kwargs: List of arguments (key, value)-pairs
        """
        if 'batch_process' not in kwargs:
            return
        batch_data = kwargs['batch_process']
        training_loss = kwargs['training_loss']
        self._find_spikes(kwargs['batch_process']["batch_data"], batch_data['batch'], kwargs["model"], training_loss)
            # for sample in batch_data['batch']:
            #     if re.search(self.inspect_rule, sample):
            #         log_msg = f"[TL = {training_loss}] Faulty sample {sample}"
            #         logger.log(
            #             self.log_level,
            #             log_msg,
            #         )
            #         break
