#
# TITLE: scientific computing
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: A Mathematical Introduction to Data Science with Python, Yi Sun et al., Springer, 2026 
#


#
# packages, sub-modules
#

import numpy as np
import sympy as sp
import sympy.stats as sps
import pandas as pd
import sklearn.datasets as sld 
import matplotlib.pyplot as plt
import scipy as sc
import statsmodels as sm


#
# switches
#

sec_1_4_sw = False
sec_1_5_sw = False
sec_1_6_sw = False
sec_1_7_sw = True


#
# 1.4 introduction to numpy
#

if sec_1_4_sw:
    # reproducible
    np.random.seed(0)
    # random integer
    m = np.random.randint(low=1, high=51, size=(20,20))
    # sort by rows
    m_sort_by_rows = m.copy()
    m_sort_by_rows.sort(axis=1)
    for each_row_index in range(m_sort_by_rows.shape[0]):
        each_row_vector = m_sort_by_rows[each_row_index,:]
        each_row_vector_max  = each_row_vector.max()
        each_row_vector_min  = each_row_vector.min()
        each_row_vector_sum  = each_row_vector.sum()
        each_row_vector_mean = each_row_vector.mean()
        #
        output_format = 'row %3i = %s (max/min/sum/mean=%i/%i/%i/%1.f)'
        output_value = tuple([each_row_index, each_row_vector, each_row_vector_max, each_row_vector_min, each_row_vector_sum, each_row_vector_mean])
        output_string = output_format % output_value
        print(output_string)
    # sort by cols
    m_sort_by_cols = m.copy()
    m_sort_by_cols.sort(axis=0)
    for each_col_index in range(m_sort_by_cols.shape[1]):
        each_col_vector = m_sort_by_cols[:,each_col_index]
        each_col_vector_max  = each_col_vector.max()
        each_col_vector_min  = each_col_vector.min()
        each_col_vector_sum  = each_col_vector.sum()
        each_col_vector_mean = each_col_vector.mean()
        #
        output_format = 'col %3i = %s (max/min/sum/mean=%i/%i/%i/%1.f)'
        output_value = tuple([each_col_index, each_col_vector, each_col_vector_max, each_col_vector_min, each_col_vector_sum, each_col_vector_mean])
        output_string = output_format % output_value
        print(output_string)

    # visualization
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(m)
    ax[1].imshow(m_sort_by_rows)
    ax[2].imshow(m_sort_by_cols)
    plt.tight_layout()
    plt.show()



#
# 1.5 introduction to sympy
#

if sec_1_5_sw:
    # set
    x = sp.FiniteSet(1,2,3,4,5)
    y = sp.FiniteSet(4,5,6,7,8)
    z = sp.FiniteSet(6,7,8,9,10)
    union = sp.Union(x, y)
    intersect = sp.Intersection(y, z)
    #
    output_format  = 'x = %s, y = %s, z = %s' + '\n'
    output_format += 'union(x,y) = %s, intersect(y,z) = %s'
    output_value = tuple([x, y, z, union, intersect])
    output_string = output_format % output_value
    print(output_string)

    # differentiation
    x = sp.symbols('x')
    f = sp.sin(x) * sp.exp(x) * sp.ln(x)
    df_dx = sp.diff(f, x)
    #
    output_format = 'f = %s, f\' = %s'
    output_value = tuple([f, df_dx])
    output_string = output_format % output_value
    print(output_string)

    # statistics
    a, b = sp.symbols('a b')
    x = sps.Uniform('x', a, b)
    y = sps.Uniform('y', -a, a)
    mean_x, mean_y = sps.E(x), sps.E(y)
    #
    output_format = 'mean_x = %s, mean_y = %s'
    output_value = tuple([mean_x, mean_y])
    output_string = output_format % output_value
    print(output_string)



#
# sec 1.6 data manipulation with pandas
#

if sec_1_6_sw:
    #
    dataset_url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    dataset_url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    col_names1 = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df1 = pd.read_csv(dataset_url1, header=None, names=col_names1)
    df2 = pd.read_csv(dataset_url2, sep=';')
    
    #
    dataset1_shape,     dataset2_shape     = df1.shape,   df2.shape
    dataset1_head,      dataset2_head      = df1.head(),  df2.head()
    dataset1_row_index, dataset2_row_index = df1.index,   df2.index
    dataset1_col_index, dataset2_col_index = df1.columns, df2.columns
    dataset1_loc = df1.loc[3:4, ['sepal_width', 'sepal_length']]
    dataset1_iloc, dataset2_iloc = df1.iloc[3:5, 1:3], df2.iloc[11:16, 1:5]
    dataset2_desc = df2.describe()
    dataset2_quality = df2['quality']
    dataset2_quality_max = dataset2_quality.max()
    dataset2_quality_cnt = dataset2_quality.value_counts()
    #
    output_format  = 'shape (1/2) = %s/%s' + '\n'
    output_format += 'head 1 =\n%s' + 'head 2=\n%s' + '\n'
    output_format += 'rows = %s' + '\n'
    output_format += 'cols = %s' + '\n'
    output_format += 'selected loc =\n%s' + '\n'
    output_format += 'selected iloc1 =\n%s' + '\n'
    output_format += 'selected iloc2 =\n%s' + '\n'
    output_format += 'description 2 =\n%s' + '\n'
    output_format += 'quality max = %s' + '\n'
    output_format += 'value_counts =\n%s'
    output_value   = [dataset1_shape, dataset2_shape, dataset1_head, dataset2_head]
    output_value  += [dataset1_row_index, dataset1_col_index, dataset1_loc, dataset1_iloc, dataset2_iloc]
    output_value  += [dataset2_desc, dataset2_quality_max, dataset2_quality_cnt]
    output_value   = tuple(output_value)
    output_string = output_format % output_value
    print(output_string)
    


#
# 1.7 machine learning with scikit-learn
#

if sec_1_7_sw == True:
    #
    dataset1 = sld.load_iris()



