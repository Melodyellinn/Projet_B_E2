import unittest
import pandas as pd
def data_read(path):
    df = pd.read_csv(path)
    return df


class test_unit_dataframe(unittest.TestCase):
    
    def test_non_null_data(self):
        path_test = "data/new_data.csv"
        df = pd.read_csv(path_test)
        assert(len(df)!=0)
    
    def test_columns_number(self):
        path_test = "data/new_data.csv"
        df = pd.read_csv(path_test)
        assert(len(df.columns)==13)
        
if __name__ == '__main__':
    unittest.main()
