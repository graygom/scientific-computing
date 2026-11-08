#
# TITLE: scientific computing
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: A Mathematical Introduction to Data Science with Python, Yi Sun et al., Springer, 2026 
#


import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

#
# switches
#

sec_1_4_sw = True



#
# 1.4 introduction to numpy
#

if sec_1_4_sw:
    # reproducible
    np.random.seed(0)
    # random integer
    m = np.random.randint(low=1, high=51, size=(10,10))
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
        output_format = 'row %3i = %s'
        output_value = tuple([each_row_index, each_row_vector])
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
        output_format = 'col %3i = %s'
        output_value = tuple([each_col_index, each_col_vector])
        output_string = output_format % output_value
        print(output_string)

    # visualization
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(m)
    ax[1].imshow(m_sort_by_rows)
    ax[2].imshow(m_sort_by_cols)
    plt.tight_layout()
    plt.show()


