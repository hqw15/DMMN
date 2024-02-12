import numpy as np
import csv
import os
import utils.define as define_utils


def write_head(writer, title):
    writer.write(
        "*****************************************************************\n")
    writer.write(
        f"                            {title}                             \n")
    writer.write(
        "*****************************************************************\n")


def write_item(writer, text):
    writer.write(f"\n > {text}                             \n")


def write_dict(writer, text_dict):
    v_str = str(text_dict)
    if 'pvalue_fdr' in text_dict:
        v_str = f"pvalue_fdr: {text_dict['pvalue_fdr']}  {str(text_dict)}"
    writer.write(f"\t{text_dict['name']} : {v_str}\n")


def get_data_by_name(info_mat, header, name):
    assert info_mat.shape[1] == len(header)
    data = info_mat[:, header.index(name)]
    data = [int(v) for v in data]
    return data


def split_group_by_fma(data, score, no_data=False):
    if no_data:
        data = np.zeros_like(score)
    assert len(data) == len(score)
    data1, data2 = list(), list()
    label = list()
    for d, s in zip(data, score):
        # 现在不需要分组
        if s < -80:
            data1.append(d)
            label.append(0)
        else:
            data2.append(d)
            label.append(1)
    # print(f'split: <80 : {len(data1)}, >=80 : {len(data2)}')
    return np.array(data1), np.array(data2), np.array(label)


class DataReader:

    def __init__(self, filter=True):
        self.filter = filter
        self.node_num = define_utils.node_num
        self.sub_dir = define_utils.sub_dir
        self.exp_dir = define_utils.exp_dir
        self.patient_info_file = define_utils.patient_info_file
        self.healthy_info_file = define_utils.healthy_info_file
        self.network_path = define_utils.network_path
        self.people_nums = define_utils.people_nums

        self._read_network_info()
        self._read_base_info()

    def _read_base_info(self, filter=True):
        self.patient_info = list()
        with open(self.patient_info_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, row in enumerate(csv_reader):
                if len(row[0].strip()) == 0:
                    continue
                if i == 0:
                    self.patient_info_header = [x.strip() for x in row[2:]]
                else:
                    self.patient_info.append([int(x) for x in row[2:]])
        if filter:
            self.patient_info = self._filter_data(self.patient_info, 'P')
        self.patient_info = np.array(self.patient_info)

        self.healthy_info = list()
        with open(self.healthy_info_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, row in enumerate(csv_reader):
                if len(row[0].strip()) == 0:
                    continue
                if i == 0:
                    self.healthy_info_header = row[1:]
                else:
                    self.healthy_info.append([int(x) for x in row[1:]])
        if filter:
            self.healthy_info = self._filter_data(self.healthy_info, 'H')
        self.healthy_info = np.array(self.healthy_info)

    def get_score(self, name):
        return self.patient_info[:, self.patient_info_header.index(name)]

    def get_before_score(self, name):
        rename = name + '-before'
        return self.patient_info[:, self.patient_info_header.index(rename)]

    def get_after_score(self, name):
        rename = name + '-after'
        return self.patient_info[:, self.patient_info_header.index(rename)]

    def get_diff_score(self, name):
        before_score = self.get_before_score(name)
        after_score = self.get_after_score(name)
        return after_score - before_score

    def get_network_mean_data(self, data_dict):
        for k in data_dict:
            assert data_dict[k].shape[1] == self.node_num
        res_dict = dict()
        for n in self.network:
            res_dict[n] = dict()
            for k in data_dict:
                data = np.zeros_like(data_dict[k][:, 0])
                for node in self.network[n]:
                    data += data_dict[k][:, node - 1]
                res_dict[n][k] = data / len(self.network[n])
        return res_dict

    def get_corr_h(self):
        age = get_data_by_name(self.healthy_info, self.healthy_info_header,
                               'age')
        sex = get_data_by_name(self.healthy_info, self.healthy_info_header,
                               'sex')
        return {'age': age, 'sex': sex}

    def get_corr_p(self):
        age = get_data_by_name(self.patient_info, self.patient_info_header,
                               'age')
        sex = get_data_by_name(self.patient_info, self.patient_info_header,
                               'sex')
        return {'age': age, 'sex': sex}

    def _read_numpy(self, path):
        return np.load(path)

    def _read_network_info(self):
        self.network = {
            'Sensorimotor System': [],
            'Visual System': [],
            'Attention System': [],
            'Default Mode System': [],
            'Subcortical  System': []
        }
        lines = open(self.network_path, 'r').readlines()
        curr_network = None
        node_list = list()
        for line in lines:
            line = line.strip()
            if line in self.network:
                curr_network = line
                continue
            node_id = int(line.split(' ')[0])
            self.network[curr_network].append(node_id)
            node_list.append(node_id)
        assert len(node_list) == len(set(node_list))
        assert np.min(node_list) == 1
        assert np.max(node_list) == len(node_list)

    def read_one_dim_data(self, file_path, data_type=''):
        lines = open(file_path, 'r').readlines()
        data = [float(line.strip()) for line in lines]
        return data

    def read_two_dim_data(self, file_path, convert_numpy=True, data_type=''):
        """N x T: N:节点数； T:时间点数量"""
        lines = open(file_path, 'r').readlines()
        data = list()
        for line in lines:
            line = line.strip().split(" ")
            remove_black_line = list()
            for v in line:
                if len(v):
                    remove_black_line.append(float(v))
            data.append(remove_black_line)

        if convert_numpy:
            data = np.array(data)

        return data

    def read_all_one_dim_data(self, name):
        data_dict = dict()
        for sd in self.sub_dir:
            data_path = os.path.join(f'{self.exp_dir}', f'{name}_{sd}.txt')
            data = self.read_one_dim_data(data_path, filter=True, data_type=sd)
            data_dict[sd] = data
        return data_dict

    def read_all_two_dim_data(self, name):
        data_dict = dict()
        for sd in self.sub_dir:
            data_path = os.path.join(f'{self.exp_dir}', f'{name}_{sd}.txt')
            data = self.read_two_dim_data(data_path, filter=True, data_type=sd)
            data_dict[sd] = data
        return data_dict
