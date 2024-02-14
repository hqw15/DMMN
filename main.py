import os
import utils.define as define_utils
from utils.common import DataReader
from tools.global_metric import GlobalMetrics
from tools.network_metric import NetworkMetrics


class Metric:

    def __init__(self):
        self.exp_dir = define_utils.exp_dir
        self.sub_dir = define_utils.sub_dir
        self.node_num = define_utils.node_num

        self.reader = DataReader()
        self.logging = True

        self._mkdir('metrics_txt')

        self._global_metrics = GlobalMetrics(is_logging=self.logging,
                                             logging_writer=open(
                                                 'metrics_txt/模块化信息.txt', 'w'),
                                             reader=self.reader)

        self._network_metrics = NetworkMetrics(is_logging=self.logging,
                                               logging_writer=open(
                                                   'metrics_txt/网络信息.txt',
                                                   'w'),
                                               reader=self.reader)

    def _mkdir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def measure_global_metric(self, name):
        self._global_metrics(name)

    def measure_network_metric(self, name):
        self._network_metrics(name)


if __name__ == '__main__':

    metric = Metric()

    metric.measure_global_metric('modularity')
    metric.measure_global_metric('community_num')
    metric.measure_global_metric('community_size')
    metric.measure_global_metric('stationarity')

    metric.measure_network_metric('flexibility')
    metric.measure_network_metric('recruitment')
    metric.measure_network_metric('integration')
