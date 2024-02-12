import os
import glob
import numpy as np
from data_reader import DataReader
from data_writer import DataWriter


class PreProcesser:

    def __init__(self):
        self.res_dir = '../result/50_90'
        self.save_dir = 'preprocess_result'
        self.exp_id = self.res_dir.split('/')[-1]
        self.num_of_exp = int(self.exp_id.split('_')[0])
        self.node_num = int(self.exp_id.split('_')[1])

        self.sub_dir = ['H', 'P']

        self.reader = DataReader()
        self.writer = DataWriter()

        self._mkdir(self.save_dir)
        self._mkdir(os.path.join(self.save_dir, self.exp_id))

    def _mkdir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def _compute_community(self, network_result):
        """
        network_result: N x T: 节点数 x 时间点
        """
        times = network_result.shape[1]
        assert network_result.shape[0] == self.node_num

        community = network_result.flatten().tolist()
        community_set = set(community)
        # 社区数量
        community_num = len(community_set)
        # 社区大小
        community_size = len(community) / len(community_set)
        # 平稳性
        stationarity_count = dict()
        for c in community_set:
            stationarity_count[c] = list()
        for t in range(times):
            c_set = set(network_result[:, t].tolist())  # 去重 当前时间点的社区
            for c in c_set:
                node_list = np.where(
                    network_result[:, t] == c)[0].tolist()  # 当前时刻t, 社区为c的结点集合
                stationarity_count[c].append(node_list)

        all_stationarity = list()
        for c in stationarity_count:
            u_count = 0
            delta_t = len(stationarity_count[c]) - 1
            if delta_t == 0:
                all_stationarity.append(1)
                continue
            for i in range(delta_t):
                node_list_1 = stationarity_count[c][i + 1]
                node_list_2 = stationarity_count[c][i]
                union = set(node_list_1).union(set(node_list_2))
                inter = set(node_list_1).intersection(set(node_list_2))
                u_count += len(inter) / len(union)
            u_count /= delta_t
            all_stationarity.append(u_count)
        # 平稳性
        stationarity = np.mean(all_stationarity)

        return community_num, community_size, stationarity

    def _compute_allegiance(self, network_result):
        """
        network_result: N x T: 节点数 x 时间点
        """
        times = network_result.shape[1]
        assert network_result.shape[0] == self.node_num

        matrix = np.zeros([self.node_num, self.node_num])

        for i in range(self.node_num):
            for j in range(i + 1, self.node_num):
                same_times = (
                    network_result[i, :] == network_result[j, :]).sum()
                matrix[i, j] = matrix[j, i] = same_times
        matrix /= times
        for i in range(self.node_num):
            matrix[i, i] = 1.0

        return matrix

    def _compute_recruitment_integration(self, allegiance_mat):
        assert allegiance_mat.shape[0] == 90
        assert allegiance_mat.shape[1] == 90

        recruitment = list()  # 招募
        integration = list()
        network = self.reader.network
        for i in range(90):
            in_sum = 0
            in_net = None
            for net in network:
                if i + 1 in network[net]:
                    in_net = network[net]
                    for j in in_net:
                        in_sum += allegiance_mat[i, j - 1]
            out_sum = np.sum(allegiance_mat[i]) - in_sum

            recruitment.append(in_sum / len(in_net))
            integration.append(out_sum / (90 - len(in_net)))

        return np.array(recruitment), np.array(integration)

    def preprocess_modularity(self, file_name='Q_all_subjects.txt'):
        for sd in self.sub_dir:
            path = os.path.join(self.res_dir, sd, file_name)
            self.writer.write_one_dim_list(
                data=self.reader.read_one_dim_data(path),
                path=f'{self.save_dir}/{self.exp_id}/modularity_{sd}.txt')

    def preprocess_flexibility(self,
                               file_name='switching_rates_all_subjects.txt'):
        for sd in self.sub_dir:
            path = os.path.join(self.res_dir, sd, file_name)
            self.writer.write_two_dim_list(
                data=self.reader.read_two_dim_data(path),
                path=f'{self.save_dir}/{self.exp_id}/flexibility_{sd}.txt')

    def preprocess_main_metric(self):
        for sd in self.sub_dir:
            res_dir = os.path.join(self.res_dir, sd, 'result')
            people_list = list(set(os.listdir(res_dir)) - set(['.DS_Store']))
            print(sd, ' people num: ', len(people_list))

            community_num_res = list()
            community_size_res = list()
            stationarity_res = list()

            allegiance_matrix_res = list()
            recruitment_res = list()  # 招募
            integration_res = list()  # 整合

            people_list.sort()  # 一定要对人按照名字排序
            for people in people_list:
                print(people)
                exp_result_list = glob.glob(
                    os.path.join(res_dir, people, '*.txt'))
                assert len(exp_result_list) == self.num_of_exp
                # 每个人的结果需要按照实验次数求均值
                community_num_list = list()
                community_size_list = list()
                stationarity_list = list()

                allegiance_matrix = np.zeros([self.node_num, self.node_num])
                recruitment_list = list()  # 招募
                integration_list = list()  # 整合

                for _, exp_result_file in enumerate(exp_result_list):
                    # 多层网络结果矩阵
                    network_result = self.reader.read_two_dim_data(
                        exp_result_file)
                    assert network_result.shape[0] == self.node_num
                    # 社区数量、大小、平稳性
                    community_num, community_size, stationarity = self._compute_community(
                        network_result)
                    community_num_list.append(community_num)
                    community_size_list.append(community_size)
                    stationarity_list.append(stationarity)
                    # 模块忠诚度矩阵
                    allegiance = self._compute_allegiance(network_result)
                    allegiance_matrix += allegiance
                    # 招募系数 整合系数
                    recruitment, integration = self._compute_recruitment_integration(
                        allegiance)
                    recruitment_list.append(recruitment)
                    integration_list.append(integration)

                # 实验次数平均
                community_num_res.append(np.mean(community_num_list))
                community_size_res.append(np.mean(community_size_list))
                stationarity_res.append(np.mean(stationarity_list))

                allegiance_matrix_res.append(allegiance_matrix /
                                             self.num_of_exp)
                # 每个人的节点水平的模块度值
                alle_mat = allegiance_matrix / self.num_of_exp
                self.writer.write_two_dim_list(
                    data=alle_mat,
                    path=
                    f'{self.save_dir}/{self.exp_id}/allegiance_matrix/{sd}_{people}.txt'
                )

                recruitment_res.append(np.mean(recruitment_list, axis=0))
                integration_res.append(np.mean(integration_list, axis=0))

            # 存储
            self.writer.write_one_dim_list(
                data=community_num_res,
                path=f'{self.save_dir}/{self.exp_id}/community_num_{sd}.txt')
            self.writer.write_one_dim_list(
                data=community_size_res,
                path=f'{self.save_dir}/{self.exp_id}/community_size_{sd}.txt')
            self.writer.write_one_dim_list(
                data=stationarity_res,
                path=f'{self.save_dir}/{self.exp_id}/stationarity_{sd}.txt')
            self.writer.write_two_dim_list(
                data=recruitment_res,
                path=f'{self.save_dir}/{self.exp_id}/recruitment_{sd}.txt')
            self.writer.write_two_dim_list(
                data=integration_res,
                path=f'{self.save_dir}/{self.exp_id}/integration_{sd}.txt')


if __name__ == '__main__':

    processer = PreProcesser()

    processer.preprocess_modularity()
    processer.preprocess_flexibility()
    processer.preprocess_main_metric()
