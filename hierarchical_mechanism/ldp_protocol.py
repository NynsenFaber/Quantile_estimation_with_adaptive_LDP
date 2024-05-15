from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer
from pure_ldp.frequency_oracles.direct_encoding import DEClient, DEServer
from pure_ldp.frequency_oracles.hadamard_response import HadamardResponseClient, HadamardResponseServer
from pure_ldp.frequency_oracles.unary_encoding import UEClient, UEServer
from .data_structure import Tree
import numpy as np
from bisect import bisect_left
import tqdm


def ldp_protocol(data: list,
                 eps: float,
                 tree: Tree,
                 replacement: bool,
                 protocol: str,
                 verbose: bool = False) -> list[LHServer]:
    """
    LDP protocol functions for the b-ary mechanism. It returns a list of servers with the privatized data for the
    b-adic decomposition of the domain (in intervals).

    Ref: Graham Cormode, Samuel Maddock, and Carsten Maple.
         Frequency Estimation under Local Differential Privacy. PVLDB, 14(11): 2046 - 2058, 2021

    GitHub: https://github.com/Samuel-Maddock/pure-LDP

    :param data: a list of data
    :param eps: privacy parameter
    :param tree: the tree structure
    :param replacement: with (True) / without (False) replacement
    :param protocol: the protocol to use
    :param verbose: print the progress bar
    :return:
    """
    data = list(data)
    intervals = tree.intervals

    clients = []
    servers = []
    levels = tree.depth
    counts = np.zeros(levels, dtype=int)
    # create the clients and servers
    for level in range(levels):
        # ------------- Local Hashing
        if protocol == 'local_hashing':
            clients.append(LHClient(epsilon=eps, d=len(intervals[level]), use_olh=True))
            servers.append(LHServer(epsilon=eps, d=len(intervals[level]), use_olh=True))

        # ------------- Direct Encoding
        elif protocol == 'direct_encoding':
            clients.append(DEClient(epsilon=eps, d=len(intervals[level])))
            servers.append(DEServer(epsilon=eps, d=len(intervals[level])))

        # ------------- Hadamard Response
        elif protocol == 'hadamard_response':
            server = HadamardResponseServer(epsilon=eps, d=len(intervals[level]))
            client = HadamardResponseClient(epsilon=eps, d=len(intervals[level]), hash_funcs=server.get_hash_funcs())
            servers.append(server)
            clients.append(client)

        # ------------- Unary Encoding
        elif protocol == 'unary_encoding':
            clients.append(UEClient(epsilon=eps, d=len(intervals[level]), use_oue=True))
            servers.append(UEServer(epsilon=eps, d=len(intervals[level]), use_oue=True))

        else:
            raise ValueError(
                f"Protocol {protocol} not recognized, try 'local_hashing', 'direct_encoding' or 'hadamard_response'"
            )

    if verbose:
        iterator = tqdm.tqdm(range(len(data)))
    else:
        iterator = range(len(data))
    for _ in iterator:
        # sample a user
        user = sample_element(data, replacement)
        # select a random level
        level = np.random.randint(0, levels)
        # get the intervals at the random level
        selected_intervals = intervals[level]
        # select the index of the subinterval where the user belongs
        interval_index = find_interval_index(selected_intervals, user)
        # get the client and server
        client = clients[level]
        # privatize the data and send to the server
        priv_data = client.privatise(interval_index)
        servers[level].aggregate(priv_data)
        counts[level] += 1

    return servers, counts


def sample_element(D: list, replacement: bool) -> int:
    """
    Sample an element from the dataset D
    :param D: dataset
    :param replacement: with (True) / without (False) replacement
    :return: the sampled element
    """
    if replacement:
        return np.random.choice(D, 1)[0]
    else:
        return D.pop(np.random.randint(0, len(D)))


def find_interval_index(interval: list[list], y: int) -> int:
    """
    Find the index of the subinterval where y belongs
    """
    for i, subinterval in enumerate(interval):
        index = bisect_left(subinterval, y)
        if index < len(subinterval) and subinterval[index] == y:
            return i
    return None