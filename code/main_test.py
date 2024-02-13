import unittest
from unittest.mock import patch
import pandas as pd
from main import validation


class MainTest(unittest.TestCase):

    @patch("main.bq_query")
    def test_validation_ok(self, mock_bq_query):
        data = {'flag': [True]}
        df_test = pd.DataFrame(data=data)
        mock_bq_query().to_dataframe.return_value = df_test
        query_file = 'file/path/mock.sql'
        resp = validation(query_file)
        self.assertTrue(resp)

    @patch("main.bq_query")
    def test_validation_error(self, mock_bq_query):
        data = {'flag': [False]}
        df_test = pd.DataFrame(data=data)
        mock_bq_query().to_dataframe.return_value = df_test
        query_file = 'file/path/mock.sql'
        resp = validation(query_file)
        self.assertFalse(resp)


if __name__ == '__main__':
    unittest.main()
