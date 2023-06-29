"""
QciClient
Utility class for user interactions with QCI API
"""

from dataclasses import dataclass
from datetime import datetime
import gzip
from itertools import chain
from io import BytesIO
import json
from posixpath import join
import time
from typing import Generator, List, Optional, Union

import networkx as nx
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from requests_futures.sessions import FuturesSession
import scipy.sparse as sp

from qci_client.base import BaseApi, BACKOFF_FACTOR, RETRY_TOTAL, STATUS_FORCELIST
from qci_client.data_converter import (
    build_graph_data_from_get_file,
    data_to_json,
    multipart_file,
)
from qci_client.utils import _buffered_generator

# TODO: temporary - for QCIEN-1142 (remove after we stop using verify=False on requests
from requests.packages.urllib3.exceptions import (  # pylint: disable=import-error, no-member, ungrouped-imports, wrong-import-order
    InsecureRequestWarning,
)

requests.packages.urllib3.disable_warnings(  # pylint: disable=no-member
    InsecureRequestWarning
)


class JobStatus:  # pylint: disable=too-few-public-methods
    """Allowed jobs statuses."""

    QUEUED = "QUEUED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


FINAL_STATUSES = frozenset([JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.CANCELLED])


@dataclass
class QciClient(BaseApi):  # pylint: disable=too-many-public-methods
    """
    Provides requests for QCIs public API as well as utility functions for creating requests and processing entire jobs.

    :param max_workers: int, number of threads for concurrent file download calls
    :param files: url path fragment to specify files API endpoint
    :param jobs: url path fragment to specify jobs API endpoint
    :param _supported_job_types: list of job_types accepted by jobs endpoint
    """

    max_workers: int = 8
    files: str = "files"
    jobs: str = "jobs"
    _supported_job_types: List[str] = None

    def __post_init__(self):
        super().__post_init__()

        # if not provided fill with qatalyst job_types (cf. qphoton)
        self._supported_job_types = (
            self._supported_job_types
            if self._supported_job_types is not None
            else frozenset(
                [
                    "sample-qubo",
                    "bipartite-community-detection",
                    "unipartite-community-detection",
                    "graph-partitioning",
                    "sample-constraint",
                    "sample-lagrange-optimization",
                    "sample-hamiltonian",
                ]
            )
        )

    @property
    def jobs_url(self):
        """Get jobs URL."""
        return join(self.url, self.jobs)

    def get_job_type_url(self, job_type: str) -> str:
        """Get job URL with job type."""
        return join(self.jobs_url, job_type)

    def get_job_type_job_id_url(self, job_type: str, job_id: str) -> str:
        """Get job URL with job type and job ID."""
        return join(self.get_job_type_url(job_type), job_id)

    def get_job_id_url(self, job_id: str) -> str:
        """Get job URL with job ID."""
        return join(self.jobs_url, job_id)

    def get_job_statuses_url(self, job_id: str) -> str:
        """Get job status using job ID."""
        return join(self.get_job_id_url(job_id), "statuses")

    def get_job_metrics_url(self, job_id: str) -> str:
        """Get job metrics using job ID."""
        return join(self.get_job_id_url(job_id), "metrics")

    def get_job_metrics_provider_url(self, job_id: str) -> str:
        """Getjob metrics for provider using job ID."""
        return join(self.get_job_metrics_url(job_id), "provider")

    @property
    def files_url(self):
        """Get files URL."""
        return join(self.url, self.files)

    def get_file_id_url(self, file_id: str) -> str:
        """Get file URL with file ID."""
        return join(self.files_url, file_id)

    def get_file_contents_url(self, file_id: str, part_num: int) -> str:
        """Get file contents URL with file ID and file part number."""
        return join(self.get_file_id_url(file_id), "contents", str(part_num))

    @BaseApi.refresh_token
    def upload_file(  # pylint: disable=too-many-branches, too-many-locals, too-many-statements
        self,
        data: Union[dict, np.ndarray, nx.Graph, sp.spmatrix, list],
        file_type: Optional[str] = None,
        file_name: Optional[str] = None,
        compress: Optional[bool] = False,
    ) -> dict:
        """
        Uploads either a formatted dict that can be uploaded directly or if supplied
        with a np.ndarray, nx.Graph, sp.spmatrix attempts to create formatted json. Must
        include file_type paramter must be provided if using any format other than dict
        for your upload.

        :param data: either a formatted dict which will compose upload body or a python
            object which will be used to create the data field in the formatted request
            body.
        :param file_type: str used in tandem with non-dict objects to create the request
            body. Optional if uploading a formatted dict which will compose the request
            body. Must be one of ["qubo", "hamiltonian", "graph", "objective",
            "constraints", "rhs"].
        :param file_name: Optional str provided by user for identification of the file.

        :return: dict with a single entry {"file_id": "some_file_id"}.

        Any open requests should be terminated promptly upon exit, say, due to
        an exception.
        """

        start_time_s = time.perf_counter()

        if self.debug:
            print(
                f"Uploading file (file_type={file_type}, file_name={file_name}, "
                f"max_workers={self.max_workers})..."
            )

        if not isinstance(data, dict):
            if self.debug:
                print(f"Converting data for file_type={file_type} to JSON...")
                print(f"\tElapsed time = {time.perf_counter() - start_time_s} s.")

            data = data_to_json(
                data=data, file_type=file_type, file_name=file_name, debug=self.debug
            )

            if self.debug:
                print(f"Converting data for file_type={file_type} to JSON...done.")
                print(f"\tElapsed time = {time.perf_counter() - start_time_s} s.")

        # Get a generator for the multipart file chunks
        multipart_gen = multipart_file(data, compress)

        if self.debug:
            print("Got multi generator.")
            print(f"\tElapsed time = {time.perf_counter() - start_time_s} s.")

        # create the zip headers with the gzip content-encoding
        if compress:
            content_encoding = {"Content-Encoding": "gzip"}
            zip_headers = self.merge_two_dicts(
                self.headers_without_connection_close, content_encoding
            )

        with FuturesSession(max_workers=self.max_workers) as session:
            max_retries = Retry(
                total=RETRY_TOTAL,
                backoff_factor=BACKOFF_FACTOR,
                status_forcelist=STATUS_FORCELIST,
            )
            session.mount("https://", HTTPAdapter(max_retries=max_retries))

            # Upload the first chunk to get a file_id from POST call.
            chunk, part_num = next(multipart_gen)

            try:
                # As with subsequent generator, wait indefinitely for future resolution.
                # However, the finite timeout on the request should be sufficient for
                # eventual termination.
                if compress:
                    zip_chunk = self.zip_payload(chunk)
                    first_response = session.request(
                        "POST",
                        join(self.url, "files"),
                        headers=zip_headers,
                        data=zip_chunk,
                        timeout=self.timeout,
                        verify=False,
                    ).result()

                else:
                    first_response = session.request(
                        "POST",
                        join(self.url, "files"),
                        headers=self.headers_without_connection_close,
                        json=chunk,
                        timeout=self.timeout,
                        verify=False,
                    ).result()

                try:
                    self._check_response_error(response=first_response)
                except AssertionError as exc:
                    raise AssertionError(
                        f"Failed to upload part {part_num} of file with name "
                        f"{file_name} and type {data['file_type']} with API "
                        f"response '{exc}'."
                    ) from exc
            except Exception as exc:
                print(f"Upload of file part {part_num} generated exception '{exc}'.")
                raise

            # Convert response to json after future resolves.
            response = first_response.json()
            file_id = response["file_id"]

            if self.debug:
                print(f"file_id = {file_id}.")
                print(f"Uploaded chunk {part_num}.")
                print(f"\tElapsed time = {time.perf_counter() - start_time_s} s.")

            # Prepare (kw)args for concurrent chunk uploads.
            if compress:
                args_kwargs_generator = (
                    (
                        (
                            "PATCH",
                            join(self.url, "files", file_id, "contents", str(part_num)),
                        ),
                        {
                            "headers": zip_headers,
                            "timeout": self.timeout,
                            "data": self.zip_payload(chunk),
                            "verify": False,
                        },
                    )
                    for chunk, part_num in multipart_gen
                )
            else:
                args_kwargs_generator = (
                    (
                        (
                            "PATCH",
                            join(self.url, "files", file_id, "contents", str(part_num)),
                        ),
                        {
                            "headers": self.headers_without_connection_close,
                            "timeout": self.timeout,
                            "json": chunk,
                            "verify": False,
                        },
                    )
                    for chunk, part_num in multipart_gen
                )

            # Upload subsequent chunks concurrently. _buffered_generator may raise
            # exception and waits indefinitely for future resolution. However, the
            # finite timeout on the request should be sufficient for eventual
            # termination.
            try:
                upload_generator = _buffered_generator(
                    buffer_length=self.max_workers,
                    task=session.request,
                    args_kwargs_generator=args_kwargs_generator,
                )
                for generated_response in upload_generator:
                    try:
                        part_num += 1
                        try:
                            self._check_response_error(response=generated_response)
                        except AssertionError as exc:
                            raise AssertionError(
                                f"Failed to upload part {part_num} of file with ID "
                                f"{file_id} and type {data['file_type']} with API "
                                f"response '{exc}'."
                            ) from exc

                        if self.debug:
                            print(f"Uploaded chunk {part_num}.")
                            print(
                                f"\tElapsed time = {time.perf_counter() - start_time_s} s."
                            )
                    except Exception as exc1:
                        # Cancel tasks and close generator.
                        try:
                            upload_generator.throw(Exception)
                        except StopIteration:
                            # Generator closed gracefully after being thrown an
                            # Exception, so report only original exception.
                            raise exc1  # pylint: disable=raise-missing-from
                        except Exception as exc2:
                            # Generator did not close gracefully after being thrown an
                            # Exception, so report both exceptions.
                            raise exc2 from exc1

                        print(
                            "Buffered generator for file upload did not close as "
                            "expected after exception in yield."
                        )
                        raise
            except Exception as exc:
                print(f"Upload of file part {part_num} generated exception '{exc}'.")
                raise

            if self.debug:
                print(
                    f"Uploading file (file_type={file_type}, file_name={file_name}, "
                    f"max_workers={self.max_workers})...done."
                )
                print(f"\tElapsed time = {time.perf_counter() - start_time_s} s.")

        return response

    @BaseApi.refresh_token
    def submit_job(self, json_dict: dict, job_type: str) -> dict:
        """
        Submit a job via a request to QCI public API.

        Args:
            json_dict: formatted json body that includes all parameters for the job
            job_type: one of the _supported_job_types

        Returns:
            Response from POST call to API
        """
        self.validate_job_type(job_type=job_type)
        response = self.session.request(
            "POST",
            self.get_job_type_url(job_type),
            json=json_dict,
            headers=self.headers,
            timeout=self.timeout,
            verify=False,
        )
        self._check_response_error(response=response)
        return response.json()

    @BaseApi.refresh_token
    def get_job_status(self, job_id: str) -> dict:
        """
        Get the status of a job by its ID.

        Args:
            job_id: ID of job

        Returns:
            Response from GET call to API
        """
        response = self.session.request(
            "GET",
            self.get_job_statuses_url(job_id),
            headers=self.headers,
            timeout=self.timeout,
            verify=False,
        )

        self._check_response_error(response=response)
        return response.json()

    @BaseApi.refresh_token
    def get_job_response(self, job_id: str, job_type: str) -> dict:
        """
        Get a response for a job by id and type, which may/may not be finished.

        :param job_id: ID of job
        :param job_type: type of job, one of []

        :return dict: loaded json file
        """
        self.validate_job_type(job_type=job_type)
        response = self.session.request(
            "GET",
            self.get_job_type_job_id_url(job_type, job_id),
            headers=self.headers,
            timeout=self.timeout,
            verify=False,
        )

        self._check_response_error(response=response)
        return response.json()

    @BaseApi.refresh_token
    def get_file(self, file_id: str, metadata: dict = None) -> Generator:
        """
        Return file contents as a generator. Each entry will be a dictionary of the form

            {
            'file_name': 'some_name', 'file_type': 'qubo', 'num_variables': 'n',
            'part_num': part number, 'data': [{'i': i, 'j': j, 'val: val}, ..., {...}]
            }

        :param file_id: str, a file id
        :param metadata: dict, metadata obtained from 'get_metadata'

        :return: generator
        """

        start_time_s = time.perf_counter()

        # Optimization to allow not fetching metadata multiple times.
        if metadata is None:
            metadata = self.get_file_metadata(file_id)

        num_parts = metadata["num_parts"]  # Not present in unsupported legacy files.
        next_part = 0

        if self.debug:
            print(f"Downloading file with ID {file_id} with {num_parts} parts...")

        with FuturesSession(max_workers=self.max_workers) as session:
            max_retries = Retry(
                total=RETRY_TOTAL,
                backoff_factor=BACKOFF_FACTOR,
                status_forcelist=STATUS_FORCELIST,
            )
            session.mount("https://", HTTPAdapter(max_retries=max_retries))

            # Prepare (kw)args for concurrent chunk downloads.
            args_kwargs_generator = (
                (
                    ("GET", self.get_file_contents_url(file_id, part_num)),
                    {
                        "headers": self.headers_without_connection_close,
                        "timeout": self.timeout,
                        "verify": False,
                    },
                )
                for part_num in range(num_parts)
            )

            # Download chunks.
            try:
                # Download chunks concurrently. _buffered_generator may raise exception
                # and waits indefinitely for future resolution. However, the finite
                # timeout on the request should be sufficient for eventual termination.
                download_generator = _buffered_generator(
                    buffer_length=self.max_workers,
                    task=session.request,
                    args_kwargs_generator=args_kwargs_generator,
                )

                for generated_response in download_generator:
                    try:
                        try:
                            self._check_response_error(response=generated_response)
                        except AssertionError as exc:
                            raise AssertionError(
                                f"Failed to download part {next_part} of file with ID "
                                f"{file_id} with API response '{exc}'."
                            ) from exc

                        if self.debug:
                            print(f"Downloaded file part {next_part}.")
                            print(
                                f"\tElapsed time = {time.perf_counter() - start_time_s} s."
                            )

                        response_json = generated_response.json()
                        # Capture next_part before yielding response JSON.
                        next_part = response_json["next_part"]
                        yield response_json
                    except Exception as exc1:
                        # Cancel tasks and close generator.
                        try:
                            download_generator.throw(Exception)
                        except StopIteration:
                            # Generator closed gracefully after being thrown an
                            # Exception, so report only original exception.
                            raise exc1  # pylint: disable=raise-missing-from
                        except Exception as exc2:
                            # Generator did not close gracefully after being thrown an
                            # Exception, so report both exceptions.
                            raise exc2 from exc1

                        print(
                            "Buffered generator for file download did not close as "
                            "expected after exception in yield."
                        )
                        raise
            except Exception as exc:
                print(f"Download of file part {next_part} generated exception '{exc}'.")
                raise

        if self.debug:
            print(f"Downloading file with ID {file_id} with {num_parts} parts...done.")
            print(f"\tElapsed time = {time.perf_counter() - start_time_s} s.")

    @BaseApi.refresh_token
    def get_file_whole(self, file_id: str) -> dict:
        """
        Return file content with complete data field. Extends get_file and combines the
        iterator portion into one data list.
            {
            'file_name': 'some_name', 'file_type': 'qubo', 'num_variables': 'n',
            'data': [{'i': i, 'j': j, 'val: val}, ..., {...}]
            }

        :param file_id: str, a file id

        :return: dict, file content with metadata
        """
        metadata = self.get_file_metadata(file_id)
        file_parts = self.get_file(file_id, metadata=metadata)
        # populate with metadata, will overwrite data field below
        if "graph" in metadata["file_type"] or "job_results" in metadata["file_type"]:
            file = build_graph_data_from_get_file(file_parts, metadata["file_type"])
        else:
            file = next(file_parts)
            file["data"].extend(list(chain(*[part["data"] for part in file_parts])))
            file.pop("next_part")

        return file

    @BaseApi.refresh_token
    def get_file_metadata(self, file_id: str) -> dict:
        """
        Get file info. Use to determine file type and dimensions for matrix construction.

        Args:
            file_id: str, from {'file_id': file_id} returned by a GET call

        Returns:
            Dictionary of metadata
        """

        if self.debug:
            print(f"Getting metadata for file with ID {file_id}...")

        start_time_s = time.perf_counter()

        response = self.session.request(
            "GET",
            self.get_file_id_url(file_id),
            headers=self.headers,
            timeout=self.timeout,
            verify=False,
        )

        stop_time_s = time.perf_counter()

        if self.debug:
            print(f"Getting metadata for file with ID {file_id}...done.")
            print(f"\tElapsed time: {stop_time_s - start_time_s} s.")

        self._check_response_error(response=response)

        return response.json()

    def validate_job_type(self, job_type: str) -> None:
        """
        Checks if a provided job type is a supported job type.

        Args:
            job_type: a job type to validate

        Returns:
            None

        Raises AssertionError if job_type is not one of the _supported_job_types.
        """
        assert (
            job_type in self._supported_job_types
        ), f"Provided job_type '{job_type}' is not one of {self._supported_job_types}"

    def build_job_body(  # pylint: disable=too-many-arguments
        self,
        job_type: str,
        qubo_file_id: Optional[str] = None,
        graph_file_id: Optional[str] = None,
        hamiltonian_file_id: Optional[str] = None,
        objective_file_id: Optional[str] = None,
        constraints_file_id: Optional[str] = None,
        rhs_file_id: Optional[str] = None,
        job_params: Optional[dict] = None,
        job_name: str = "job_0",
        job_tags: Optional[list] = None,
    ) -> dict:
        """
        Constructs body for job submission requests
        :param job_type: one of _supported_job_types
        :param qubo_file_id: file id from files API for uploaded qubo
        :param graph_file_id: file id from files API for uploaded graph
        :param hamiltonian_file_id: file id from files API for uploaded hamiltonian
        :param objective_file_id: file id from files API for uploaded objective
        :param constraints_file_id: file id from files API for uploaded constraints
        :param rhs_file_id: file id from files API for uploaded rhs
        :param job_params: dict of additional params to be passed to job submission in "params" key
        :param job_name: user specified name for job submission
        :param job_tags: user specified labels for classifying and filtering user jobs after submission
        :note: Need to add validation for job parameters
        """
        if job_params is None:
            job_params = {}

        if job_tags is None:
            job_tags = []

        self.validate_job_type(job_type=job_type)
        assert (
            sum(
                fid is not None
                for fid in [
                    qubo_file_id,
                    graph_file_id,
                    hamiltonian_file_id,
                    objective_file_id,
                ]
            )
            == 1
        ), "Only one of qubo_file_id, hamiltonian_file_id, objective_file_id, or graph_file_id can be specified"
        job_body = {"job_name": job_name, "job_tags": job_tags}
        if job_type == "sample-constraint":
            assert None not in [
                objective_file_id,
                constraints_file_id,
                rhs_file_id,
            ], "objective_file_id, constraints_file_id, and rhs_file_id must all be specified for job_type='sample-constraint'"
            job_body.update(
                {
                    "objective_file_id": objective_file_id,
                    "constraints_file_id": constraints_file_id,
                    "rhs_file_id": rhs_file_id,
                }
            )
        else:
            assert all(
                fid is None
                for fid in [constraints_file_id, rhs_file_id, objective_file_id]
            ), "objective_file_id, constraints_file_id, and rhs_file_id are not available for selected job_type"
            if job_type == "sample-qubo":
                assert (
                    qubo_file_id is not None
                ), "qubo_file_id must be specified for job_type='sample-qubo'"
                job_body["qubo_file_id"] = qubo_file_id
            elif job_type == "sample-hamiltonian":
                assert (
                    hamiltonian_file_id is not None
                ), "hamiltonian_file_id must be specified for job_type='sample-hamiltonian'"
                job_body["hamiltonian_file_id"] = hamiltonian_file_id
            else:
                # if add more jobs that are non graph would need to change this else
                assert (
                    graph_file_id is not None
                ), "graph_file_id must be specified for the given job_type"
                job_body["graph_file_id"] = graph_file_id

        # list of job_params vary by job_type
        # could add a validate params function here
        job_body.update({"params": job_params})
        return job_body

    def print_job_log(self, message: str) -> None:
        """
        Formats a messages for updating user with a time stamp appended
        :param message: a string to be passed in print statement
        """
        print(f"{message}: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

    # TODO: add provider and device names args (optional)

    def process_job(self, job_type: str, job_body: dict, wait: bool = True) -> dict:
        """
        :param job_type: the type of job being processed must be one of _supported_job_types
        :param job_body: formatted json dict for body of job submission request
        :param wait: bool indicating whether or not user wants to wait for job to complete

        :return:
            if wait is True, then dict with job_info response and results file
                (results is None if ERROR or CANCELLED)
            if wait is False, then response dict from submitted job, which includes job
                ID for subsequent retrieval
        :note: what else do we want to return with the results? response_id, obviously job_id
        """
        self.validate_job_type(job_type=job_type)
        submit_response = self.submit_job(json_dict=job_body, job_type=job_type)
        job_id = submit_response["job_id"]
        self.print_job_log(message=f"Job submitted job_id='{job_id}'-")
        curr_status = None
        if wait:
            while curr_status not in FINAL_STATUSES:
                time.sleep(1)
                status_response = self.get_job_status(job_id=job_id)
                iter_status = status_response["status"]
                if iter_status != curr_status:
                    self.print_job_log(message=iter_status)
                    curr_status = iter_status

            job_response = self.get_job_response(job_id=job_id, job_type=job_type)
            if curr_status in [JobStatus.CANCELLED, JobStatus.ERROR]:
                results = None
            else:
                results_fid = job_response["results"]["file_id"]
                results = self.get_file_whole(file_id=results_fid)
            return {"job_info": job_response, "results": results}

        return submit_response

    @BaseApi.refresh_token
    def list_files(self, username: Optional[str] = None) -> dict:
        """
        :param username: Optional str - username (to search for files owned by the named user)
            mostly useful when run by users with administrator privileges (such as QCI users) who can see all files.
            When called by an administrator, the username parameter is used to restrict the list files returned
            to be only the files owned by the user specified in the username parameter.
            When run by non-privileged users, this parameter is truly optional because non-privileged users
            will only ever see lists of files that they created.

        :return: dict containing list of files
        """
        if username:
            querystring = {"regname": "username", "regvalue": username}

            response = self.session.request(
                "GET",
                self.files_url,
                headers=self.headers,
                params=querystring,
                timeout=self.timeout,
                verify=False,
            )
        else:
            response = self.session.request(
                "GET",
                self.files_url,
                headers=self.headers,
                timeout=self.timeout,
                verify=False,
            )

        self._check_response_error(response=response)
        return response.json()

    @BaseApi.refresh_token
    def delete_file(self, file_id: str) -> dict:
        """
        :param file_id: str - file_id of file to be deleted

        :return: dict containing information about file deleted (or error)
        """

        if self.debug:
            print(f"Deleting file with ID {file_id}...")

        start_time_s = time.perf_counter()

        response = self.session.request(
            "DELETE",
            self.get_file_id_url(file_id),
            headers=self.headers,
            timeout=self.timeout,
            verify=False,
        )

        stop_time_s = time.perf_counter()

        if self.debug:
            print(f"Deleting file with ID {file_id}...done.")
            print(f"\tElapsed time: {stop_time_s - start_time_s} s.")

        self._check_response_error(response=response)

        return response.json()

    @BaseApi.refresh_token
    def zip_payload(self, payload: str) -> bytes:
        """
        :param payload: str - json contents of file to be zipped

        "return: zipped request_body
        """
        out = BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w", compresslevel=6) as file:
            file.write(json.dumps(payload).encode("utf-8"))
        request_body = out.getvalue()
        out.close()
        return request_body

    @BaseApi.refresh_token
    def merge_two_dicts(self, x, y):  # pylint: disable=invalid-name
        """Given two dictionaries, merge them into a new dict as a shallow copy."""
        z = x.copy()  # pylint: disable=invalid-name
        z.update(y)
        return z

    def get_job_type_from_job_id(self, job_id: str) -> str:
        """
        Get job type from job ID.

        Args:
            job_id: ID of the job

        Returns:
            Type of the job
        """
        response_job_metadata_short = self.session.request(
            "GET",
            self.get_job_id_url(job_id),
            headers=self.headers,
            timeout=self.timeout,
            verify=False,
        )
        self._check_response_error(response=response_job_metadata_short)

        return response_job_metadata_short.json()["type"].replace("_", "-")
