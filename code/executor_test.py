import unittest
from unittest.mock import patch, MagicMock
from executor import ddl_executor, dml_executor, _executor, _tuple_executor, \
    _create_thread


class MainTest(unittest.TestCase):

    @patch("bq_process.create_table_bq")
    def test_ddl_executor(self, mock_create_table_bq):
        list_task = [{'file': 'archivo1',
                      'table': 'tabla1',
                      'dataset': 'dataset1'},
                     {'file': 'archivo2',
                      'table': 'tabla2',
                      'dataset': 'dataset2'}]
        ddl_executor(list_task)
        self.assertTrue(mock_create_table_bq.called)

    @patch("executor._executor")
    @patch("bq_process.validation_step")
    def test_dml_executor_task(self, mock_validation_step, mock_executor):
        mock_validation_step.return_value = False
        task = [{'file': 'test'}]
        dml_executor(task)
        self.assertTrue(mock_executor.called)

    @patch("executor._tuple_executor")
    @patch("bq_process.validation_step")
    def test_dml_executor_tuple(self, mock_validation_step, mock_tuple):
        mock_validation_step.return_value = False
        task = [({'file': 'test_1'}, {'file': 'test_2'})]
        dml_executor(task)
        self.assertTrue(mock_tuple.called)

    @patch("executor._executor")
    def test_dml_executor_error(self, mock_executor):
        mock_executor.return_value = False
        task = [{'file': 'test_1'}]
        with self.assertRaises(Exception) as context:
            dml_executor(task)
            self.assertTrue('Query Error' in context.exception)

    @patch("executor._create_thread")
    def test_dml_executor_thread(self, mock_create_thread):
        task = [[{'file': 'test_1'}, {'file': 'test_2'}]]
        dml_executor(task)
        self.assertTrue(mock_create_thread.called)

    @patch("bq_process.validation_step")
    def test_executor_skip_step(self, mock_validation_step):
        mock_validation_step.return_value = True
        task = {'file': 'task_1',
                'dataset': 'TEST_TT_DATA',
                'table': 'table_test',
                'type': 'WRITE_APPEND',
                }
        resp = _executor(task)
        self.assertTrue(resp)

    @patch("bq_process.validation_step")
    @patch("bq_process.insert_table")
    @patch("bq_process.update_status")
    def test_executor_step_ok_write(self, mock_update_status,
                                    mock_insert_table,
                                    mock_validation_step):
        mock_update_status.return_value = True
        mock_insert_table.return_value = True
        mock_validation_step.return_value = False
        task = {'file': 'task_1',
                'dataset': 'TEST_TT_DATA',
                'table': 'table_test',
                'type': 'WRITE_APPEND',
                }
        resp = _executor(task)
        mock_insert_table.assert_called_with('TEST_TT_DATA', 'table_test',
                                             'WRITE_APPEND', 'resources/dml/task_1')
        mock_validation_step.assert_called_with('task_1')
        mock_update_status.assert_called_with('task_1', 'Ok')
        self.assertTrue(resp)

    @patch("bq_process.validation_step")
    @patch("bq_process.bq_query")
    @patch("bq_process.update_status")
    def test_executor_step_ok_delete(self, mock_update_status,
                                     mock_bq_query, mock_validation_step):
        mock_update_status.return_value = True
        mock_bq_query.return_value = True
        mock_validation_step.return_value = False
        task = {'file': 'task_2',
                'dataset': 'TEST_TT_DATA',
                'table': 'table_test',
                'type': 'DELETE',
                }
        resp = _executor(task)
        mock_bq_query.assert_called_with('resources/dml/task_2')
        mock_validation_step.assert_called_with('task_2')
        mock_update_status.assert_called_with('task_2', 'Ok')
        self.assertTrue(resp)

    @patch("bq_process.validation_step")
    @patch("bq_process.bq_query")
    @patch("bq_process.update_status")
    def test_executor_step_error(self, mock_update_status,
                                 mock_bq_query, mock_validation_step):
        mock_update_status.return_value = True
        mock_bq_query.side_effect = Exception()
        mock_validation_step.return_value = False
        task = {'file': 'task_3',
                'dataset': 'TEST_TT_DATA',
                'table': 'table_test',
                'type': 'DELETE',
                }
        resp = _executor(task)
        mock_bq_query.assert_called_with('resources/dml/task_3')
        mock_validation_step.assert_called_with('task_3')
        mock_update_status.assert_called_with('task_3', 'Error')
        self.assertFalse(resp)

    @patch("executor._executor")
    def test_tuple_executor_ok(self, mock_executor):
        mock_executor.return_value = True
        task = ({'file': 'test_1'}, {'file': 'test_2'})
        _tuple_executor(task)
        mock_executor.assert_has_calls(mock_executor.call({'file': 'test_1'}),
                                       mock_executor.call({'file': 'test_2'}))

    @patch("executor._executor")
    def test_tuple_executor_error(self, mock_executor):
        mock_executor.return_value = False
        task = ({'file': 'test_1'}, {'file': 'test_2'})
        with self.assertRaises(Exception) as context:
            _tuple_executor(task)
            self.assertTrue('Query Error' in context.exception)
        mock_executor.assert_has_calls(mock_executor.call({'file': 'test_1'}),
                                       mock_executor.call({'file': 'test_2'}))

    @patch("executor._create_thread")
    def test_tuple_executor_thread(self, mock_create_thread):
        mock_create_thread.return_value = True
        task = ([{'file': 'test_1'}, {'file': 'test_2'}],)
        _tuple_executor(task)
        mock_create_thread.assert_called_with([{'file': 'test_1'},
                                               {'file': 'test_2'}])

    @patch("multiprocessing.dummy.Pool")
    @patch("executor.dml_executor")
    def test_create_thread(self, mock_dml_executor,  mock_thread_pool):
        tasks = [[{'file': 'test'}, {'file': 'test2'}]]
        mock_dml_executor.return_value = True
        _create_thread(tasks)
        mock_thread_pool(2).return_value = MagicMock()
        self.assertTrue(mock_thread_pool.called)


if __name__ == '__main__':
    unittest.main()
