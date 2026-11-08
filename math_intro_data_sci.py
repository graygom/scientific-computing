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
# 1.4 numpy
#

if True:
    #
    np.random.seed(0)
    #
    m = np.random.randint(low=1, high=51, size=(20,20))
    #
    m_sort_by_rows = m.copy()
    m_sort_by_rows.sort(axis=1)
    #
    m_sort_by_cols = m.copy()
    m_sort_by_cols.sort(axis=0)

    #
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(m)
    ax[1].imshow(m_sort_by_rows)
    ax[2].imshow(m_sort_by_cols)
    plt.tight_layout()
    plt.show()
