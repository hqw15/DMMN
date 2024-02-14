import glob
import numpy as np
import utils.define as utils_define
from utils.common import DataReader


def ge50(x):
    return x >= 50


def l50(x):
    return x < 50


class AllegianceMat:

    def __init__(self):

        self.sub_dir = utils_define.sub_dir
        self.exp_dir = utils_define.exp_dir
        self.node_num = utils_define.node_num
        self.reader = DataReader()

        print(self.reader.get_before_score('FMA'))

    def ReadMat(self, data_type='P', fun=ge50, name=''):
        assert data_type in self.sub_dir
        fma = self.reader.get_before_score('FMA')
        print(len(fma))
        mat_list = list()
        data_paths = glob.glob(
            f"{self.exp_dir}/allegiance_matrix/{data_type}_*.txt")
        data_paths.sort()

        for idx, p in enumerate(data_paths):
            if data_type == 'P' and fun(fma[idx - 1]):
                continue
            mat = self.reader.read_two_dim_data(p, convert_numpy=True)
            assert mat.shape[0] == self.node_num
            assert mat.shape[1] == self.node_num
            mat_list.append(mat)

        mat_list = np.array(mat_list)
        # 所有人平均
        mat_array = np.mean(mat_list, axis=0)

        network_id = {
            0: 'Sensorimotor System',
            1: 'Visual System',
            2: 'Attention System',
            3: 'Default Mode System',
            4: 'Subcortical  System'
        }
        network_num = len(network_id)
        final_mat = np.zeros([network_num, network_num])
        for i in range(network_num):
            for j in range(network_num):
                n1 = network_id[i]
                n2 = network_id[j]
                idx1_list = self.reader.network[n1]
                idx2_list = self.reader.network[n2]
                for idx1 in idx1_list:
                    for idx2 in idx2_list:
                        final_mat[i, j] += mat_array[idx1 - 1, idx2 - 1]
                final_mat[i, j] /= (len(idx1_list) * len(idx2_list))
        print(name)
        print(final_mat)


if __name__ == "__main__":

    p = AllegianceMat()
    p.ReadMat('H', name='健康人')
    p.ReadMat('P', ge50, name='>=50')
    p.ReadMat('P', l50, name='<50')
