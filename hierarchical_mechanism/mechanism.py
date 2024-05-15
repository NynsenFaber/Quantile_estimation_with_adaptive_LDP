from .data_structure import Tree
from .ldp_protocol import ldp_protocol


def hierarchical_mechanism_quantile(tree: Tree,
                                    data: list,
                                    protocol: str,
                                    eps: float,
                                    target: float,
                                    replacement: bool = False) -> int:
    """
    Hierarchical mechanism specialized for quantile estimation.

    Ref: Kulkarni, Tejas. "Answering range queries under local differential privacy."
         Proceedings of the 2019 International Conference on Management of Data. 2019.

    It returns the quantile of the data using the tree structure, which is computed as the closest value to the quantile
    of the cumulative distribution function of the privatized data.

    :param tree: the tree structure
    :param data: the data to privatize
    :param protocol: the protocol to use for ldp (for low privacy only unary_encoding works)
    :param eps: the privacy parameter
    :param target: the quantile or target in (0,1) to estimate
    :param replacement: with (True) / without (False) replacement
    """

    assert 0 < target < 1, "Quantile must be between 0 and 1"

    servers, counts = ldp_protocol(data=data,
                                   eps=eps,
                                   tree=tree,
                                   replacement=replacement,
                                   protocol=protocol,
                                   verbose=False)

    # update the tree using the privatized data
    tree.update_tree(servers, counts)

    # get the cdf
    tree.compute_cdf()

    # get the quantile
    coin = tree.get_quantile(target)

    return coin
