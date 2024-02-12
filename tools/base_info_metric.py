import numpy as np
import scipy.stats as stats
import utils.common as common_utils
import utils.statistics as statistics_utils
import utils.define as define_utils


class DataItem:

    def __init__(self, data, header, name):
        self._data = data
        self._header = header
        self._name = name

    def all_data(self):
        return self._data

    def header(self):
        return self._header

    def name(self):
        return self._name

    def num(self):
        return len(self._data)

    def data(self, name):
        return common_utils.get_data_by_name(self._data, self._header, name)


class BaseInfo:

    def __init__(self, is_logging, logging_writer, reader):
        self._is_logging = is_logging
        self._logging_writer = logging_writer
        self._reader = reader

        self._healthy = DataItem(reader.healthy_info,
                                 reader.healthy_info_header, '健康人')
        self._patient = DataItem(reader.patient_info,
                                 reader.patient_info_header, '病人')

        self._score_list = define_utils.base_info_scores

    def _split_patient_data(self, data):
        return common_utils.split_patient_data(data, self._reader)

    def _base_info_metric(self, item1, item2, need_other_metric=False, type='HP'):
        a_name, b_name = item1.name(), item2.name()
        corr_a, corr_b = self._reader.get_corr_h(), self._reader.get_corr_p()

        # 1. people nums
        a_nums, b_nums = item1.num(), item2.num()
        # 2. age
        age_res = statistics_utils.independent_sample_test(
            item1.data('age'),
            item2.data('age'),
            corr_a,
            corr_b,
            name_a=f'{a_name}年龄',
            name_b=f'{b_name}年龄')
        # 3. sex
        a_sexs, b_sexs = item1.data('sex'), item2.data('sex')
        a_sexs = [a_sexs.count(1), a_sexs.count(2)]
        b_sexs = [b_sexs.count(1), b_sexs.count(2)]
        sex_pvalue = stats.chi2_contingency([a_sexs, b_sexs],
                                            correction=True)[1],
        sex_res = {
            'pvalue': sex_pvalue,
            f'{a_name}性别(男女)': a_sexs,
            f'{b_name}性别(男女)': b_sexs,
            '检验方法': '卡方检验',
        }

        # 4. day and other index
        other_score_res = list()
        if need_other_metric:
            score_list = ['day']
            for s in self._score_list:
                score_list.append(s + '-before')
            for v in score_list:
                v_show = '病程' if v == 'day' else v.replace('-before', '')
                other_score_res.append([
                    v,
                    statistics_utils.independent_sample_test(
                        item1.data(v),
                        item2.data(v),
                        corr_a,
                        corr_b,
                        name_a=f'{a_name}{v_show}',
                        name_b=f'{b_name}{v_show}')
                ])

        if self._is_logging:
            common_utils.write_item(self._logging_writer,
                                    f'\t{a_name} VS {b_name}')
            self._logging_writer.write(
                f'数量 : {a_name}:{a_nums},  {b_name}:{b_nums}\n')
            self._logging_writer.write('年龄 : ' + str(age_res) + '\n')
            self._logging_writer.write('性别 : ' + str(sex_res) + '\n')
            if need_other_metric:
                for v, res in other_score_res:
                    self._logging_writer.write(f'{v} : {str(res)}\n')


    def _diff_info_metric_non_group(self):
        res= list()
        for v in self._score_list:
            before_score = self._reader.get_before_score(v)
            after_score = self._reader.get_after_score(v)
            res.append([
                v,
                statistics_utils.paired_sample_test(self._reader,
                                                    before_score,
                                                    after_score,
                                                    name_a='治疗前',
                                                    name_b='治疗后')
            ])
            
        if self._is_logging:
            common_utils.write_item(self._logging_writer, f'\t治疗前 VS 治疗后 (不分组)')
            for v, res in res:
                self._logging_writer.write(f'{v} : {str(res)}\n')

    def __call__(self):
        if self._is_logging:
            common_utils.write_head(self._logging_writer, '基本信息')

        # 1. 健康人和病人
        self._base_info_metric(self._healthy, self._patient)
        # 2. 治疗前后，不分组
        self._diff_info_metric_non_group()
