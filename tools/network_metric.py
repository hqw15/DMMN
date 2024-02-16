import os
import utils.statistics as statistics_utils
import utils.common as common_utils


class NetworkMetrics:

    def __init__(self, is_logging, logging_writer, reader):

        self._is_logging = is_logging
        self._logging_writer = logging_writer
        self._reader = reader

    def __call__(self, name):
        if self._is_logging:
            common_utils.write_head(self._logging_writer, name)

        data_dict = self._reader.read_all_two_dim_data(name)

        self._compute_network(data_dict)
        self._compute_whole_head(data_dict)

        if self._is_logging:
            self._logging_writer.write('\n')

    def _compute_network(self, data_dict):
        # 网络（脑区）水平
        network_res = list()
        network_data_dict = self._reader.get_network_mean_data(data_dict)
        for network in self._reader.network:
            network_res.append(
                statistics_utils.measure_metric(
                    dh=network_data_dict[network]['H'],
                    dp=network_data_dict[network]['P'],
                    reader=self._reader))
        if self._is_logging:
            for i, network in enumerate(self._reader.network):
                common_utils.write_item(self._logging_writer,
                                        f'\t网络: {network}')
                for k, v in network_res[i].items():
                    self._logging_writer.write(v)
            if 1:
                common_utils.write_head(self._logging_writer, '具体数据')
                network_data_dict = self._reader.get_network_mean_data(
                    data_dict)
                for network in self._reader.network:
                    common_utils.write_item(self._logging_writer,
                                            f'\t网络: {network}')
                    data = network_data_dict[network]
                    for k, v in data.items():
                        self._logging_writer.write(f'\t\t - {k}: ')
                        for i in v:
                            self._logging_writer.write(f'{i} ')
                        self._logging_writer.write('\n')

    def _compute_whole_head(self, data_dict):
        # 全脑
        head_res = statistics_utils.measure_metric(dh=data_dict['H'].mean(1),
                                                   dp=data_dict['P'].mean(1),
                                                   reader=self._reader)
        if self._is_logging:
            common_utils.write_item(self._logging_writer, '\t全脑:')
            for k, v in head_res.items():
                self._logging_writer.write(v)
