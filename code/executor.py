import logging
from multiprocessing.dummy import Pool as ThreadPool

import bq_process as bq

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(name)s] - %(message)s", level=logging.INFO
)
logger = logging


class QueryError(Exception):
    """Raise for my specific kind of exception."""


def _executor(task: dict) -> bool:
    """Run the defined task

    :param task: Dictionary with the definition of the task.
    :return: Execution status.
    """
    step_name = str(task.get("file"))
    if not bq.validation_step(step_name):
        resp = None
        try:
            if (
                task.get("type") == "WRITE_TRUNCATE"
                or task.get("type") == "WRITE_APPEND"
            ):
                resp = bq.insert_table(
                    str(task.get("dataset")),
                    str(task.get("table")),
                    str(task.get("type")),
                    "resources/dml/" + str(task.get("file")),
                )

            if task.get("type") == "DELETE" or task.get("type") == "MERGE":
                resp = bq.bq_query("resources/dml/" + str(task.get("file")))
            if not resp:
                bq.update_status(step_name, "Error")
                return False
            else:
                bq.update_status(step_name, "Ok")
            return True
        except Exception as e:
            logger.error(e)
            bq.update_status(step_name, "Error")
            return False
    else:
        logger.info(f"Skip step: {step_name}")
        return True


def _create_thread(tasks: list) -> None:
    """Generates the thread of executions for the tasks that will be
    parallelized.

    :param tasks: Tasks to execute.
    """
    args = []
    for task in tasks:
        args.append([task])
    pool = ThreadPool(len(tasks))
    pool.starmap(dml_executor, args)
    pool.close()
    pool.join()


def _tuple_executor(tuple_tasks: tuple) -> None:
    """Execution of sequential tasks.

    :param tuple_tasks: Sequential task tuple.
    """
    for task in tuple_tasks:
        if isinstance(task, dict):
            resp = _executor(task)
            if not resp:
                raise QueryError("Query Error")
        else:
            _create_thread(task)


def ddl_executor(task_list: list) -> None:
    """Create the tables in bigquery according to the definition in task_list.

    :param task_list: List of dictionaries with the tables to create.
    """
    for task in task_list:
        bq.create_table_bq(
            "resources/ddl/" + task.get("file"),
            task.get("table"),
            task.get("dataset"),
            task.get("file_type"),
            task.get("control_table"),
            task.get("clustering"),
            task.get("partition"),
            task.get("description"),
        )


def dml_executor(task_list: list) -> None:
    """Executor of tasks defined by the development teams in the file task.

    :param task_list: List of tasks to execute.
    """
    for task in task_list:
        if isinstance(task, tuple):
            _tuple_executor(task)
        elif isinstance(task, dict):
            resp = _executor(task)
            if not resp:
                raise QueryError("Query Error")
        else:
            _create_thread(task)
