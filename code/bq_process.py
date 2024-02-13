import logging
from google.cloud import bigquery
from google.cloud.bigquery import TimePartitioning, QueryJob
from google.cloud.bigquery.table import RowIterator
from google.cloud.exceptions import NotFound
import properties as props
import task

logging.basicConfig(
    format='[%(asctime)s %(levelname)s %(name)s] - %(message)s',
    level=logging.INFO)
logger = logging

frequency = {
    "daily": "CURRENT_DATE('America/Santiago')",
    "weekly": "DATE_TRUNC(CURRENT_DATE('America/Santiago'),WEEK(MONDAY))",
    "monthly": "DATE_TRUNC(CURRENT_DATE('America/Santiago'),month)",
    "full": "CURRENT_DATETIME()"
}

flag_control_table = task.tabla_control[0].get('control_table')
control_table_project_id = props.PROJECT_CONFIG_CLIENT if flag_control_table else props.PROJECT_CONFIG


def _table_exists(dataset: str, table_name: str, project_id: str) -> bool:
    """ Check if the table exists in the dataset.

    :param dataset: dataset name
    :param table_name: table name
    :param project_id: project id
    :return: True if it exists, False otherwise
    """

    client = bigquery.Client(project_id)
    table_id = f"{project_id}.{dataset}.{table_name}"
    try:
        client.get_table(table_id)
        return True
    except NotFound:
        return False


def _create_schema(metadata: str) -> list:
    """ Generate table schema in bigquery from file.

    :param metadata: Path to the schema definition file.
    :return: list with the bigquery schema.
    """
    schema = []
    with open(metadata, 'r') as dml:
        lines = dml.readlines()
        for line in lines:
            column = line.split(";")
            schema.append(bigquery.SchemaField(column[0].strip(), column[1],
                                               column[2].strip(), column[3]))
    return schema


def _truncate_table(dataset: str, table: str):
    """ Truncate table in bigquery.

        :param dataset: Name of the dataset.
        :param table: Name of the table.
        :return: google.cloud.bigquery.table.RowIterator query result.
    """
    try:
        query = f'TRUNCATE TABLE `{props.PROJECT_CONFIG}.{dataset}.{table}` '
        client = bigquery.Client()
        query_job = client.query(query)
        logger.info(f'TRUNCATE TABLE Successfully {query_job.result()}')
    except Exception as e:
        logger.error('TRUNCATE TABLE - Errors occurred during execution')
        raise e


def _create_query_from_file(file: str) -> str:
    """ Generates a string with the content of the query in the file.

    :param file: Path to the query definition file.
    :return: string with query
    """
    with open(file, 'r') as f:
        lines = f.readlines()
        query_execute = ''.join(lines).format(props.PROJECT_CONFIG)
    return query_execute


def _bq_delete_query(dataset: str, table: str, condition: str) -> RowIterator:
    """ Delete data from a table in bigquery.

    :param dataset: Name of the dataset.
    :param table: Name of the table.
    :param condition: Delete condition.
    :return: google.cloud.bigquery.table.RowIterator query result.
    """
    query = f'DELETE FROM `{props.PROJECT_CONFIG}.{dataset}.{table}` ' \
            f'WHERE {condition}'
    client = bigquery.Client()
    query_job = client.query(query)
    return query_job.result()


def validation_step(step: str) -> bool:
    """
    Validation if a step of the task flow has already been executed in the
    time window that the process has defined, to avoid executions of steps
    that ended successfully.
    :param step: Name of the step that is running.
    :return: True if the step has already had a successful execution within
     the time window. False otherwise.
    """

    process_frequency = frequency[props.FREQUENCY]
    query = f'SELECT status FROM `{control_table_project_id}.' \
            f'{props.DATASET_STG_CONFIG}.ctrl_{props.PROCESS_CONFIG}` WHERE ' \
            f'step_name="{step}" AND execution_dt=(select max(execution_dt) ' \
            f'FROM `{control_table_project_id}.{props.DATASET_STG_CONFIG}.' \
            f'ctrl_{props.PROCESS_CONFIG}` WHERE step_name="{step}" ' \
            f'AND execution_dt >={process_frequency})'
    query_job = bq_query(query, cmd=True)
    query_job = query_job.to_dataframe()

    if len(query_job.status.unique()) == 1 and \
            query_job.status.unique()[0] == 'ok':
        return True
    else:
        return False


def bq_query(query: str, cmd: bool = None) -> RowIterator:
    """ Function that allows executing a query on a file or a character
    string in Bigquery.

    :param query: Path to the query definition file or query string.
    :param cmd: True if query string.
    :return: google.cloud.bigquery.table.RowIterator query result.
    """
    if cmd:
        query_execute = query
    else:
        query_execute = _create_query_from_file(query)

    client = bigquery.Client()
    query_job = client.query(query_execute)
    return query_job.result()


def create_table_bq(metadata: str, table_name: str, dataset: str, file_type: str = None, control_table: bool = False,
                    clustering: str = None, partition: str = None, description=None):
    """Crea tabla segun la definición entregada.

    Args:
        metadata (str): Ruta del archivo con los datos del esquema.
        table_name ([type]): Nombre de la tabla a crear.
        dataset (str): Nombre del Dataset donde se creara la tabla.
        file_type (str, opcional): Tipo de archivo del schema.
        control_table (bool): Indica si se trata de una tabla de control del proceso o tabla de datos
        partition (str, optional): Campo por el cual será particionada.
        Defaults to ''.
        clustering (str, optional): Campo por el cual será clusterizada.
        Defaults to ''.
        description (str, optional): Descripcion para asignar a la tabla.
        Defaults to ''.

    Raises:
        e: Error en la generación de la tabla.
    """

    # configure project according to the type of table to create
    if control_table:
        project_id = props.PROJECT_CONFIG_CLIENT
    else:
        project_id = props.PROJECT_CONFIG

    if _table_exists(dataset, table_name, project_id):
        logger.info(f'TABLE CREATION - Table {table_name} already exists')
    else:
        try:
            client = bigquery.Client(props.PROJECT_CONFIG_CLIENT)

            if file_type == 'json':
                schema = client.schema_from_json(metadata)
            else:
                schema = _create_schema(metadata)

            dataset_ref = bigquery.DatasetReference(
                project=project_id,
                dataset_id=dataset)
            table_ref = dataset_ref.table(table_name.lower())
            table = bigquery.Table(table_ref, schema=schema)
            if partition is not None:
                table.time_partitioning = TimePartitioning(
                    field=str(partition))

            if clustering is not None:
                clustering_list = clustering.split(',')
                table.clustering_fields = clustering_list

            if description is not None:
                table.description = description

            table = client.create_table(table)

            logger.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

        except Exception as e:

            logger.error(f"TABLE CREATION - Ocurrieron errores durante la ejecucion de la "
                         f"tarea para la tabla {table_name}")
            raise e


def _get_info_job_bq(query_job: QueryJob) -> None:
    logger.info(f"Job ID: {query_job.job_id}")
    logger.info(f"Start Date: {query_job.started}, End Date: {query_job.ended}")
    logger.info(f"Query executed successfully: \n {query_job.query}")
    logger.info(f"Billing tier: \n {query_job.billing_tier}")
    logger.info(f"Total bytes billed: \n {query_job.total_bytes_billed}")
    logger.info(f"Total bytes processed: \n {query_job.total_bytes_processed}")


def insert_table(dataset: str, table_name: str, write_disposition: str,
                 query: str) -> bool:
    """ Inserts data into the destination table.

    :param dataset: Name of the dataset where the table will be created.
    :param table_name: Name of the table to create.
    :param write_disposition: Specifies the action that occurs if the
                             destination table already exists.
    :param query: Path to the query definition file.
    :return: True if the query was executed successfully, False otherwise.
    """
    if not _table_exists(dataset, table_name, props.PROJECT_CONFIG):
        logger.warning(f'Errors occurred during execution - The Table: '
                       f' {table_name} does not exist.')
        return False
    else:
        execute_query = _create_query_from_file(query)
        try:
            client = bigquery.Client(props.PROJECT_CONFIG_CLIENT)
            if write_disposition == 'WRITE_TRUNCATE':
                # WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the table data and uses the schema
                # from the query result.
                # To maintain the original schema of the table, data deletion is performed through TRUNCATE TABLE
                _truncate_table(dataset, table_name)

            write_disposition = bigquery.WriteDisposition.WRITE_APPEND

            job_config = bigquery.QueryJobConfig(
                write_disposition=write_disposition,
                destination=bigquery.DatasetReference(
                    project=props.PROJECT_CONFIG,
                    dataset_id=dataset)
                .table(table_name.lower())
            )
            query_job = client.query(execute_query, location='US',
                                     job_config=job_config)

            query_job.result()
            #query_job.add_done_callback(_get_info_job_bq)

            return True
        except Exception as e:
            logger.error(f'INSERT TABLE - Errors occurred during the '
                         f'execution of the task for the table:{table_name}')
            logger.error(e)
            raise e


def update_status(step: str, status: str) -> None:
    """ Update status in control table.
    :param step: Step name.
    :param status:  Ok, Error.

    """

    logger.info(f"PROJECT_ID: {control_table_project_id} of control table")

    query = f"INSERT INTO `{control_table_project_id}." \
            f"{props.DATASET_STG_CONFIG}.ctrl_{props.PROCESS_CONFIG}`" \
            f" values('{step}', '{status}', " \
            f"CURRENT_DATETIME('America/Santiago'))"
    bq_query(query, cmd=True)
