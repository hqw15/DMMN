import numpy as np


def three(fma, dh, dp, corr_h, corr_p):
    res_total = ''
    from scipy.stats import kstest, kruskal
    import statsmodels.api as sm
    import pandas as pd
    group1 = dh
    age_group1 = corr_h['age']
    group2, age_group2 = [], []
    group3, age_group3 = [], []
    T = 50
    for idx, i in enumerate(fma):
        if i >= T:
            group2.append(dp[idx])
            age_group2.append(corr_p['age'][idx])
        else:
            group3.append(dp[idx])
            age_group3.append(corr_p['age'][idx])

    # 定义一个函数，用于执行 Kolmogorov-Smirnov 检验
    def normality_test(data):
        _, p_value = kstest(data, 'norm')
        return p_value

    # 定义一个函数，用于执行 Kruskal-Wallis H 检验
    def kruskal_wallis_test(*args):
        _, p_value = kruskal(*args)
        return p_value

    # 执行 Kolmogorov-Smirnov 检验评估数据的正态性
    p_value_group1 = normality_test(group1)
    p_value_group2 = normality_test(group2)
    p_value_group3 = normality_test(group3)

    # 如果数据不符合正态分布，则执行 Kruskal-Wallis H 检验
    if any(p_value < 0.05
           for p_value in [p_value_group1, p_value_group2, p_value_group3]):
        res_total += ("数据不符合正态分布，执行 Kruskal-Wallis H 检验：\n")
        p_value_kw = kruskal_wallis_test(group1, group2, group3)
        res_total += (f"Kruskal-Wallis H 检验 p 值: {p_value_kw} \n")

        z1 = np.percentile(group1, 50)
        z2 = np.percentile(group1, 75)
        z3 = np.percentile(group1, 25)
        res_total += (f"健康人: 中位数 {z1}, 上四分位数 {z2}, 下四分位数 {z3}\n")
        z1 = np.percentile(group2, 50)
        z2 = np.percentile(group2, 75)
        z3 = np.percentile(group2, 25)
        res_total += (f">=50: 中位数 {z1}, 上四分位数 {z2}, 下四分位数 {z3}\n")
        z1 = np.percentile(group3, 50)
        z2 = np.percentile(group3, 75)
        z3 = np.percentile(group3, 25)
        res_total += (f"<50: 中位数 {z1}, 上四分位数 {z2}, 下四分位数 {z3}\n")

    else:
        res_total += ("数据符合正态分布，执行 ANCOVA 分析：\n")
        groups_data = [group1, group2, group3]
        ages_data = [age_group1, age_group2, age_group3]
        # 合并年龄数据
        ages_combined = np.concatenate(ages_data)
        # 合并组数据
        groups_combined = np.concatenate(groups_data)
        # 构建设计矩阵
        X = sm.add_constant(ages_combined)
        # 拟合模型
        model = sm.OLS(groups_combined, X)
        result = model.fit()
        # 打印回归结果
        res_total += (str(result.summary()) + '\n')

        z1 = np.mean(group1)
        z2 = np.std(group1)
        res_total += (f"健康人: 均值 {z1}, 标准差 {z2}\n")
        z1 = np.mean(group2)
        z2 = np.std(group2)
        res_total += (f">=50: 均值 {z1}, 标准差 {z2}\n")
        z1 = np.mean(group3)
        z2 = np.std(group3)
        res_total += (f"<50: 均值 {z1}, 标准差 {z2}\n")

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    res_total += ("进行 Tukey's HSD test")
    groups_data = np.concatenate([group1, group2, group3])
    groups = ['健康人'] * len(group1) + ['>=50'] * len(group2) + ['<50'
                                                               ] * len(group3)
    ages_data = [age_group1, age_group2, age_group3]

    tukey = pairwise_tukeyhsd(endog=groups_data, groups=groups, alpha=0.05)
    res_total += (str(tukey) + "\n")
    return res_total


def measure_metric(dh, dp, reader):
    res = three(reader.get_before_score('FMA'), dh, dp, reader.get_corr_h(),
                reader.get_corr_p())
    return {'res': res}
