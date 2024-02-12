import glob
import numpy as np
from data_reader import DataReader


def ge50(x):
    return x >= 50


def l50(x):
    return x < 50


class AllegianceMat:

    def __init__(self):

        self.reader = DataReader()

    def ReadMat(self, data_type='P', fun=ge50, name=''):
        assert data_type in ('P', 'H')
        fma = self.reader.get_before_score('FMA')
        mat_list = list()
        data_paths = glob.glob(
            f"preprocess_result/50_90/allegiance_matrix/{data_type}_*.txt")
        data_paths.sort()

        for p in data_paths:
            idx = int(p.split("/")[-1].split('.')[0][-2:])
            if data_type == 'P' and fun(fma[idx - 1]):
                continue
            mat = self.reader.read_two_dim_data(p, convert_numpy=True)
            assert mat.shape[0] == 90
            assert mat.shape[1] == 90
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
        final_mat = np.zeros([5, 5])
        for i in range(5):
            for j in range(5):
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
