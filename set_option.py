import pandas as pd

def set_pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 500)
    # çıktının tek bir satırda olmasını sağlar.
    pd.set_option('display.expand_frame_repr', False)
    # max_rows almamızı sağlar
    pd.set_option('display.max_rows', None)
