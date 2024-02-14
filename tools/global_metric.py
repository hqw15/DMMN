import numpy as np
import utils.statistics as statistics_utils
import utils.common as common_utils


class GlobalMetrics:

    def __init__(self, is_logging, logging_writer, reader):

        self._is_logging = is_logging
        self._logging_writer = logging_writer
        self._reader = reader

    def __call__(self, name):
        if self._is_logging:
            common_utils.write_head(self._logging_writer, name)

        data_dict = self._reader.read_all_one_dim_data(name)
        metric_res = statistics_utils.measure_metric(dh=data_dict['H'],
                                                     dp=data_dict['P'],
                                                     reader=self._reader)
        if self._is_logging:
            for k, v in metric_res.items():
                self._logging_writer.write(v)
            self._logging_writer.write('\n')
        return metric_res
