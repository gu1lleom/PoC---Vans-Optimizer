import unittest
from unittest.mock import patch, MagicMock, Mock, PropertyMock, mock_open
import pandas as pd
from google.cloud import bigquery
import properties as props
from bq_process import _table_exists, _create_schema, _bq_delete_query, \
    _create_query_from_file, validation_step, bq_query, create_table_bq, \
    insert_table, update_status
import task
fake_file_path = 'file/path/mock'
query = 'file/path/query'
fake_query = 'Select 1'
fake_metadata = 'file/path/metadata'


class MainTest(unittest.TestCase):

    @patch("bq_process.bigquery.Client")
    def test_table_exists(self, mock_bq_client):
        dataset = 'dataset1'
        table_name = 'table1'
        project_id = 'project1'
        mock_bq_client.return_value = MagicMock()
        results = Mock()
        type(results).table_exists = PropertyMock(side_effect=[True])
        mock_bq_client().get_table.return_value = [results]
        resp = _table_exists(dataset, table_name, project_id)
        self.assertTrue(resp)
        self.assertTrue(mock_bq_client.called)

    @patch('builtins.open', new_callable=mock_open, read_data='campo1;campo2'
                                                              ';campo3;campo4')
    def test_create_schema(self, mock_read):
        resp = [bigquery.SchemaField('campo1', 'campo2', 'campo3', 'campo4', (), None)]
        string_read = _create_schema(fake_file_path)
        mock_read.assert_called_with(fake_file_path, 'r')
        self.assertEqual(string_read, resp)

    @patch('builtins.open', new_callable=mock_open, read_data=fake_query)
    def test_create_query_from_file(self, mock_read):
        string_read = _create_query_from_file(fake_file_path)
        mock_read.assert_called_with(fake_file_path, 'r')
        self.assertEqual(string_read, fake_query)

    @patch("bq_process.bigquery.Client")
    def test_bq_delete_query(self, mock_bq_client):
        mock_bq_client().query.return_value.result.return_value = 'delete'
        actual = _bq_delete_query('dataset_id', 'table_id', 'condition')
        query_delete = f"DELETE FROM " \
                       f"`{props.PROJECT_CONFIG}.dataset_id.table_id` " \
                       f"WHERE condition"
        mock_bq_client().query.assert_called_with(query_delete)
        self.assertEqual(actual, 'delete')

    @patch("bq_process.bigquery.Client")
    def test_validation_step_success(self, mock_bq_client):
        d = {'status': ['ok', 'ok']}
        df = pd.DataFrame(data=d)
        mock_bq_client().query.return_value.result.return_value.to_dataframe\
            .return_value = df
        actual = validation_step('step')
        self.assertEqual(actual, True)

    @patch("bq_process.bigquery.Client")
    def test_validation_step_fail(self, mock_bq_client):
        d = {'status': ['ok', 'error']}
        df = pd.DataFrame(data=d)
        mock_bq_client().query.return_value.result.return_value.to_dataframe\
            .return_value = df
        actual = validation_step('step')
        self.assertEqual(actual, False)

    @patch("bq_process.bigquery.Client")
    def test_bq_query_cmd(self, mock_bq_client):
        mock_bq_client().query.return_value.result.return_value = 1
        actual = bq_query(fake_query, cmd=True)
        self.assertEqual(actual, 1)

    @patch("bq_process._create_query_from_file")
    @patch("bq_process.bigquery.Client")
    def test_bq_query(self, mock_bq_client, mock_create_from_file):
        mock_bq_client().query.return_value.result.return_value = 1
        mock_create_from_file.return_value = fake_query
        actual = bq_query('k')
        self.assertEqual(actual, 1)

    @patch("bq_process.create_table_bq")
    @patch("bq_process._table_exists")
    def test_create_table_bq(self, mock_create_table_bq, mock_table_exists):
        metadata = MagicMock()
        table_name = MagicMock()
        dataset = MagicMock()
        mock_table_exists.return_value = True
        create_table_bq(metadata, table_name, dataset)
        self.assertTrue(mock_create_table_bq.called)

    @patch("bq_process._create_schema")
    @patch("bq_process.bigquery.Client")
    @patch("bq_process._table_exists")
    def test_create_table_bq_false(self, mock_table_exists, mock_bq_client,
                                   mock_create_schema):

        table_name = 'table_test'
        dataset = 'dataset_test'
        partition = 'view_date'
        clustering = 'CAMPO1,CAMPO2'
        mock_create_schema.return_value = [bigquery.SchemaField('CAMPO1',
                                                                'CAMPO2',
                                                                'CAMPO3',
                                                                None,
                                                                (), None)]
        mock_bq_client().return_value = MagicMock()
        mock_table_exists.return_value = False
        create_table_bq(fake_metadata, table_name, dataset, partition,
                        clustering)
        self.assertTrue(mock_table_exists.called)

    @patch("bq_process.bigquery.Client")
    @patch("bq_process._table_exists")
    def test_create_table_bq_exception(self, mock_table_exists,
                                       mock_bq_client):
        table_name = 'table_test'
        dataset = 'dataset_test'
        mock_table_exists.return_value = False
        mock_bq_client.side_effect = Exception("Error")
        with self.assertRaises(Exception) as context:
            create_table_bq(fake_metadata, table_name, dataset)
            self.assertTrue('Error' in context.exception)
        self.assertTrue(mock_table_exists.called)

    @patch("bq_process._table_exists")
    def test_insert_table_not_exists(self, mock_table_exists):
        mock_table_exists.return_value = False
        write_disposition = 'WRITE_APPEND'
        table_name = 'table_test'
        dataset = 'dataset_test'
        resp = insert_table(dataset, table_name, write_disposition, query)
        self.assertTrue(mock_table_exists.called)
        self.assertFalse(resp)

    @patch("bq_process.bigquery.Client")
    @patch("bq_process._create_query_from_file")
    @patch("bq_process._table_exists")
    def test_insert_table_exists_append(self, mock_table_exists,
                                        mock_create_from_file, mock_bq_client):
        mock_table_exists.return_value = True
        mock_create_from_file.return_value = fake_query
        mock_bq_client('test').return_value = MagicMock()
        write_disposition = 'WRITE_APPEND'
        table_name = 'table_test'
        dataset = 'dataset_test'
        resp = insert_table(dataset, table_name, write_disposition, query)
        self.assertTrue(mock_table_exists.called)
        self.assertTrue(resp)

    @patch("bq_process.bigquery.Client")
    @patch("bq_process._create_query_from_file")
    @patch("bq_process._table_exists")
    def test_insert_table_exists_truncate(self, mock_table_exists,
                                          mock_create_from_file,
                                          mock_bq_client):
        mock_table_exists.return_value = True
        mock_create_from_file.return_value = fake_query
        mock_bq_client('test').return_value = MagicMock()
        write_disposition = 'WRITE_TRUNCATE'
        table_name = 'table_test'
        dataset = 'dataset_test'
        resp = insert_table(dataset, table_name, write_disposition, query)
        self.assertTrue(mock_table_exists.called)
        self.assertTrue(resp)

    @patch("bq_process.bigquery.Client")
    @patch("bq_process._create_query_from_file")
    @patch("bq_process._table_exists")
    def test_insert_table_exception(self, mock_table_exists,
                                    mock_query_from_file, mock_bq_client):
        mock_table_exists.return_value = True
        mock_query_from_file.return_value = fake_query
        mock_bq_client.side_effect = Exception("Error")
        dataset = 'dataset1'
        table_name = 'table1'
        write_disposition = 'WRITE_TRUNCATE'
        with self.assertRaises(Exception) as context:
            insert_table(dataset, table_name, write_disposition, query)
            self.assertTrue('Error' in context.exception)
        self.assertTrue(mock_query_from_file.called)
        self.assertTrue(mock_table_exists.called)
        self.assertTrue(mock_bq_client.called)

    @patch("bq_process.bq_query")
    def test_update_status(self, mock_bq_query):
        mock_bq_query.return_value = True
        step = 'task_1'
        status = 'Error'
        update_status(step, status)

        control_table = task.tabla_control[0]
        flag_control_table = control_table.get('control_table')
        project_id = props.PROJECT_CONFIG_CLIENT if flag_control_table else props.PROJECT_CONFIG

        query_status = f"INSERT INTO `{project_id}.{props.DATASET_STG_CONFIG}" \
                       f".ctrl_{props.PROCESS_CONFIG}` values('task_1', 'Error', " \
                       "CURRENT_DATETIME('America/Santiago'))"
        mock_bq_query.assert_called_with(query_status, cmd=True)


if __name__ == '__main__':
    unittest.main()
