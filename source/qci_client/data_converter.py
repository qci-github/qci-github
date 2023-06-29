"""Functions for data conversion."""

import json
from math import floor
import sys
import time
from typing import Generator, Optional, Union

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp

# We want to limit the memory size of each uploaded chunk to be safely below the max of 15 * MebiByte (~15MB).
# See https://git.qci-dev.com/qci-dev/qphoton-files-api/-/blob/main/service/files.go#L32.
MEMORY_MAX: int = 8 * 1000000  # 8MB


def load_json_file(file_name: str) -> dict:
    """
    Load a utf-8-encoded json file into a dictionary.

    :param file_name: name of the JSON file to load

    :return dict: loaded json file
    """
    with open(file_name, "r", encoding="utf-8") as file:
        return json.loads(file.read())


def get_size(obj, seen=None) -> int:
    """
    Recursively finds size of objects

    :param obj: data object to recursively compute size of
    :param seen: takes a set and is used in the recursive step only to record whether an object has been counted yet.

    :return int:
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)
    return size


def _get_soln_size(soln):
    # Check whether first entry is a graph node/class assignment, eg., {'id': 4, 'class': 2}
    if isinstance(soln[0], dict):
        return get_size(soln)

    return sys.getsizeof(soln[0]) * len(soln)


def compute_results_step_len(data: Union[np.ndarray, list]) -> int:
    """
    Compute the step length for "chunking" the providd data.

    Args:
        data: An numpy array or list of data

    Returns:
        The step length for "chunking" the data
    """
    # total mem size of soln vector
    soln_mem = _get_soln_size(data)
    # num_vars * step_len < 30k => step_len < 30k/num_vars
    chunk_ratio = MEMORY_MAX / soln_mem
    step_len = floor(chunk_ratio) if chunk_ratio >= 1 else 1
    return step_len


def multipart_file(data: dict, compress: bool = False) -> Generator:
    """
    Break file-to-upload's data dictionary into chunks, formatting correctly with each
    returned chunk.

    :param data: dict, object from data_to_json() function

    :return: generator of (chunk, part_num) tuples
    """
    if compress:
        # The user has chosen to compress their files for upload we want a large chunksize to try to maximize compression
        # for each of the chunks
        data_chunk_size_max = 200000
    else:
        # We are using the multipart upload as a validated sharding system that is similar to Mongo GridFS.
        # Mongo recommends 256KB for that system, this value keeps uploaded chunks below this value. After some testing,
        # we decided to limit this to chunks of 10000 elements for performance reasons.
        data_chunk_size_max = 10000
    # data field for iterating. For results data, the n^2 sized data is in the samples
    # field so chunk it up. If graph, just return the data for now.
    if "job_results" in data["file_type"]:
        data_chunk_size = compute_results_step_len(data["samples"][0])
        return _results_generator(data, data_chunk_size)

    if "graph" in data["file_type"]:
        return _graph_generator(data, data_chunk_size_max)

    return _data_generator(data, data_chunk_size_max)


def _data_generator(data: dict, step_length: int) -> Generator:
    df = data["data"]  # pylint: disable=invalid-name

    # df may be empty, so use max(1, len(df)).
    for part_num, i in enumerate(range(0, max(1, len(df)), step_length)):
        if part_num == 0:
            chunk = {key: val for key, val in data.items() if key not in ["data"]}
            chunk["data"] = df[i : i + step_length]
        else:
            chunk = {"file_type": data["file_type"], "data": df[i : i + step_length]}
        yield chunk, part_num


def _graph_generator(data: dict, step_length: int) -> Generator:
    num_links = len(data["links"])
    num_nodes = len(data["nodes"])

    # links and nodes may both be empty, so use max(1, num_links, num_nodes).
    for part_num, i in enumerate(range(0, max(1, num_links, num_nodes), step_length)):
        if part_num == 0:
            chunk = {
                key: val for key, val in data.items() if key not in ["links", "nodes"]
            }
            chunk["links"] = data["links"][i : i + step_length]
            chunk["nodes"] = data["nodes"][i : i + step_length]
        else:
            # patch requires 'file_type' and 'graph', as well as node and link chunks
            chunk = {
                "file_type": data["file_type"],
                "graph": data["graph"],
                # Lengths below are unequal, meaning that one of these will return an empty list if indices DNE
                "links": data["links"][i : i + step_length],
                "nodes": data["nodes"][i : i + step_length],
            }
        print(chunk)
        yield chunk, part_num


def _results_generator(data: dict, step_length: int) -> Generator:
    file_type = data["file_type"]
    #
    # GRAPH PARTITIONING
    #
    if file_type.split("_")[-1] == "partitioning":
        for part_num, i in enumerate(range(0, len(data["samples"]), step_length)):
            chunk = {"file_type": data["file_type"], "file_name": data["file_name"]}
            if part_num == 0:
                chunk.update(
                    {
                        key: val
                        for key, val in data.items()
                        if key
                        not in [
                            "samples",
                            "energies",
                            "counts",
                            "balance",
                            "cut_size",
                            "is_feasible",
                        ]
                    }
                )
            chunk.update(
                {
                    "samples": data["samples"][i : i + step_length],
                    "energies": data["energies"][i : i + step_length],
                    "counts": data["counts"][i : i + step_length],
                    "balance": data["balance"][i : i + step_length],
                    "cut_size": data["cut_size"][i : i + step_length],
                    "is_feasible": data["is_feasible"][i : i + step_length],
                }
            )
            yield chunk, part_num
    #
    # (UNI AND BI)-COMMUNITY DETECTION
    #
    elif file_type.split("_")[-1] == "detection":
        for part_num, i in enumerate(range(0, len(data["samples"]), step_length)):
            chunk = {"file_type": data["file_type"]}
            if part_num == 0:
                # TODO: split up counts and energies too. They are also expected by API in each upload
                chunk.update(
                    {
                        key: val
                        for key, val in data.items()
                        if key
                        not in [
                            "samples",
                            "energies",
                            "counts",
                            "modularity",
                            "is_feasible",
                        ]
                    }
                )
            chunk.update(
                {
                    "samples": data["samples"][i : i + step_length],
                    "energies": data["energies"][i : i + step_length],
                    "counts": data["counts"][i : i + step_length],
                    "modularity": data["modularity"][i : i + step_length],
                    "is_feasible": data["is_feasible"][i : i + step_length],
                }
            )
            yield chunk, part_num
    #
    # NON-GRAPH RESULTS DATA
    #
    else:
        # Number of samples should align with len(energies) and len(counts)
        for part_num, i in enumerate(range(0, len(data["samples"]), step_length)):
            chunk = {"file_type": data["file_type"]}
            if part_num == 0:
                chunk.update(
                    {
                        key: val
                        for key, val in data.items()
                        if key not in ["samples", "energies", "counts"]
                    }
                )
            chunk.update(
                {
                    "samples": data["samples"][i : i + step_length],
                    "energies": data["energies"][i : i + step_length],
                    "counts": data["counts"][i : i + step_length],
                }
            )
            yield chunk, part_num


def build_graph_data_from_get_file(file_parts: Generator, file_type: str) -> dict:
    """
    Return graph data built from a given graph file generator and type.

    Args:
        file_parts: File-part generator
        file_type: Type of the file

    Returns:
        Dictionary of graph data
    """
    # Each case needs to extract metadata from the first element off the heap
    first_part = next(file_parts)
    if "graph" == file_type:
        #
        # REBUILD A LARGE GRAPH FILE
        #
        file_data = {
            key: val for key, val in first_part.items() if key not in ["links", "nodes"]
        }
        first_part = _munge_node_link_format(first_part)
        file_data["links"] = first_part["links"]
        file_data["nodes"] = first_part["nodes"]
        for part in file_parts:
            scrubbed_data = _munge_node_link_format(part)
            file_data["links"] += scrubbed_data["links"]
            file_data["nodes"] += scrubbed_data["nodes"]
        return file_data

    if file_type.split("_")[-1] == "detection":
        #
        # COMMUNITY DETECTION RESULTS
        #
        file_data = dict(first_part.items())
        for part in file_parts:
            file_data["samples"] += part["samples"]
            file_data["energies"] += part["energies"]
            file_data["counts"] += part["counts"]
            file_data["modularity"] += part["modularity"]
            file_data["is_feasible"] += part["is_feasible"]
    elif file_type.split("_")[-1] == "partitioning":
        #
        # GRAPH PARTITIONING RESULTS
        #
        file_data = dict(first_part.items())
        for part in file_parts:
            file_data["samples"] += part["samples"]
            file_data["energies"] += part["energies"]
            file_data["counts"] += part["counts"]
            file_data["cut_size"] += part["cut_size"]
            file_data["balance"] += part["balance"]
            file_data["is_feasible"] += part["is_feasible"]
    else:
        file_data = dict(first_part.items())
        for part in file_parts:
            file_data["samples"] += part["samples"]
            file_data["energies"] += part["energies"]
            file_data["counts"] += part["counts"]
    # we don't need next_part anymore
    file_data.pop("next_part")

    return file_data


def _munge_node_link_format(data: dict) -> dict:
    """
    Temporary hack to clean incoming data. Mongo BSON is adding key/value pairs to the node_link format, such as

        'nodes': [[{'Key': 'bipartite', 'Value': 0},
                {'Key': 'id', 'Value': 'Evelyn Jefferson'}],
                [{'Key': 'bipartite', 'Value': 0}, ... ]

        This should look like

        'nodes': [{'bipartite': 0, 'id': 'Evelyn Jefferson'},
                  {'bipartite': 0, 'id': 'Laura Mandeville'},
                  {'bipartite': 0, 'id': 'Theresa Anderson'}, ... ]

    This method just munges the first format in the second format.

    Args:
        data: dict,

    Returns:
        dict, node_link graph
    """
    node_link = {}
    for element in data.keys():
        if element in ("links", "nodes"):
            node_link[element] = []
            # node may exhaust earlier than links, so we only append as needed
            if len(data[element]) > 0:
                for item_list in data[element]:
                    item_props = {item["Key"]: item["Value"] for item in item_list}
                    node_link[element].append(item_props)
    return node_link


def data_to_json(
    data: Union[np.ndarray, sp.spmatrix, nx.Graph, list],
    file_type: str,
    file_name: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """
    Converts  data input into JSON string that can be passed to Qatalyst REST API
    :param data: object to be converted to JSON
       * currently constraint_penalties require one value for each constraint
    :param file_type: one of ["graph", "qubo", "objective", "constraints", "constraint_penalties", "rhs", "hamiltonian"]
    :param file_name: Optional user specified name for file to be uploaded
    :param debug: Optional, if set to True, enables debug output (default = False for no debug output)

    :return: string (JSON format)
    :note: could add support for matrices stored as lists
    """
    supported_file_types = [
        "graph",
        "qubo",
        "hamiltonian",
        "rhs",
        "objective",
        "constraints",
        "constraint_penalties",
    ]

    result_file_types = [
        "job_results_sample_qubo",
        "job_results_sample_hamiltonian",
        "job_results_sample_constraint",
        "job_results_sample_lagrange_optimization",
        "job_results_unipartite_community_detection",
        "job_results_bipartite_community_detection",
        "job_results_graph_partitioning",
    ]

    supported_file_types += result_file_types

    start = time.perf_counter()
    assert (
        file_type in supported_file_types
    ), f"Unsupported file type input specify one of {supported_file_types}"
    file_name = f"{file_type}.json" if not file_name else file_name
    file_body = {"file_type": file_type, "file_name": file_name}
    if file_type == "graph":
        assert isinstance(
            data, nx.Graph
        ), "'graph' file_type must be type networkx.Graph"
        data = json_graph.node_link_data(data)
    else:
        assert not isinstance(data, nx.Graph), (
            "file_types ['rhs', 'objective', 'qubo', 'constraints', 'constraint_penalties', 'hamiltonian'] "
            "do not support networkx.Graph type"
        )
        if file_type in ["rhs", "constraint_penalties"]:
            if isinstance(data, sp.spmatrix):
                data = data.toarray()
            if isinstance(data, np.ndarray):
                # deal with case where shape (1,n) or (n,1)
                data = data.flatten()
                data = data.tolist()
            file_body["num_constraints"] = len(data)
        else:
            assert isinstance(data, (sp.spmatrix, np.ndarray)), (
                "file_types = ['qubo', 'objective', 'constraints', 'hamiltonian'] only support types np.ndarray "
                "and scipy.sparse.spmatrix"
            )
            data = sp.coo_matrix(data)
            rows, cols = data.shape

            data_ls = []
            for i, j, val in zip(
                data.row.tolist(), data.col.tolist(), data.data.tolist()
            ):
                data_ls.append({"i": i, "j": j, "val": val})
            # must reassign to var that will be used in final update to dict
            data = data_ls
            if file_type == "constraints":
                file_body.update({"num_constraints": rows, "num_variables": cols})
            else:
                file_body["num_variables"] = rows
        # update both rhs and all objective and constraint data key
        data = {"data": data}
    # use update because graph data has more than one entry
    file_body.update(data)

    if debug:
        print(f"Time to convert data to json: {time.perf_counter()-start} s.")

    return file_body
