import unittest

# @pytest.fixture()
# def test_data():
#     df = pd.read_csv("data/data.csv")
#     return df

# def test_df_out(df):
#     df = pd.read_csv("data/data.csv")
#     # Test if df have same number's columns
#     assert len(df.columns) == 13

class MyPythonTest(unittest.TestCase):
    # def test_data():
    #     df = pd.read_csv("data/data.csv")
    #     print("df OK")

    def setUp(self):
        print("Avant le test")

    def tearDown(self):
        print("Après le test")

    def test_simple(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()