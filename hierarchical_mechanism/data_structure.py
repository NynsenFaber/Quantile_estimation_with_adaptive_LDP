import numpy as np
import math


def fill_tree(node, branching):
    """
    Fill the tree starting from the root (then uses recursion) using the branching factor.
    :param node: a node
    :param branching: the branching factor
    """
    data = node.data
    split = max(len(data) // branching, 1)
    for _ in range(branching):
        child_indices = range(split)
        child_data = data[child_indices]
        child_node = Node(child_data)
        child_node.father = node

        if len(data) > 1:
            data = np.delete(data, child_indices)
        else:
            break

        if len(child_data) > 1:
            # recursion
            fill_tree(child_node, branching)

        node.add_child(child_node)

    # add remaining data
    if len(data > 0):
        child_node = Node(data)
        child_node.father = node
        if len(data) > 1:
            fill_tree(child_node, branching)
        node.add_child(child_node)


def get_frequency(server, count, item) -> float:
    """
    Estimate the frequency of an item using the server and the count.
    :param server: a server (an instance of LDP Frequency Oracle server of pure_ldp package)
    :param count: the count of the data (server returns absolute frequency)
    :param item: the item to estimate
    """
    return server.estimate(item, suppress_warnings=True) / count


class Node(object):
    """
    A Node of a tree.
    """

    def __init__(self, data: np.array):
        self.data = data
        self.children = []
        self.attribute = 0
        self.father = None
        self._hierarchy = np.nan  # used internally to compute the cdf

    def add_child(self, child: "Node"):
        self.children.append(child)


class Tree(Node):
    """
    Tree data structure for b-ary decomposition, used for hierarchical mechanism in LDP.

    Ref:    Kulkarni, Tejas. "Answering range queries under local differential privacy."
            Proceedings of the 2019 International Conference on Management of Data. 2019.

    It works only for len(data) = branching^depth, in such a way the b-adic decomposition is complete and simple.
    The object contains a specialized method to compute the cdf of the data fast.
    """

    def __init__(self, bins: np.array, branching: int):

        assert math.log(len(bins),
                        branching).is_integer(), "The data structure works only for bins of power of the branching"

        super().__init__(bins)
        self.branching = branching
        fill_tree(self, branching)  # fill the tree
        for level in range(0, self.get_depth()):  # add hierarchy (used for cdf computation)
            self._add_hierarchy(level)
        self.depth = self.get_depth()
        self.intervals = self._get_intervals()  # store the b-adic decomposition as list[list] (level, index)
        self.cdf = None
        # these two attributes are used to compute the cdf
        self._cum_chunks = []
        self._attribute_levels = []

    def update_tree(self, servers, counts):
        """
        Update the attribute of the nodes using the servers and the counts. The function updates also
        some inner attributes used to compute the cdf.
        :param servers: a list of servers (an instance of LDP Frequency Oracle server of pure_ldp package)
        :param counts: a list of counts (server returns absolute frequency)
        """
        depth = self.depth
        self._cum_chunks = []  # reset
        self._attribute_levels = []  # reset
        for level in range(0, depth):
            nodes = self.get_nodes_at_level(level)
            cum_chunks = []
            cum_chunks_none = []
            for i, node in enumerate(nodes):
                node.attribute = get_frequency(servers[level], counts[level], i)
                if node._hierarchy is not np.nan:
                    cum_chunks.append(node.attribute)
                    cum_chunks_none.append(node.attribute)
                else:
                    cum_chunks_none.append(np.nan)
            self._cum_chunks.append(cumulative_sum_chunks(cum_chunks, self.branching - 1))
            self._attribute_levels.append(cum_chunks_none)

    def get_depth(self) -> int:
        """
        Get the depth of the tree.
        """
        max_level = 0
        for child in self.children:
            max_level = max(max_level, self._get_depth(child))
        return max_level

    def get_nodes_at_level(self, level) -> list[Node]:
        """
        Returns all the nodes at a given level. (eg. level=0 returns the leafs, level=depth returns the root)
        """
        return self._get_nodes_at_level(self, level, self.get_depth())

    def get_range_r(self, leaf_data, S=0, verbose=False) -> float:
        """
        Get a right range query (eg, leaf_data = [4] returns the cdf of the bin 4). It is slow to compute the cdf.
        """
        return self._get_range_r(self, leaf_data, S, verbose)

    def binary_search_for_quantile(self, quantile):
        """
        It works only if the cdf estimated is monotonically increasing, which for low epsilon is not the case.
        Is fast but not accurate.
        """
        max_error = 1
        bins = self.data
        max_i = np.random.choice(bins)

        # Binary search for the CDF value
        left = 0
        right = len(bins) - 1

        while left <= right:
            mid = (left + right) // 2
            cdf = self.get_range_r([mid])  # Assuming tree.get_range_r() returns the CDF for bin i
            error = quantile - cdf
            # print(f"In mid: {mid}, cdf: {cdf}, error: {error}")

            # Update max_p if current CDF value is less than the current max_p
            if np.abs(error) < max_error:
                max_error = np.abs(error)
                max_i = mid

            # Adjust search range based on the error
            if error > 0:  # Adjust right boundary
                left = mid + 1
            else:  # Adjust left boundary
                right = mid - 1
        # if max_error > alpha:
        #     print("Error is greater than alpha")
        return max_error, max_i

    def compute_cdf(self):
        """
        Compute the whole cdf of the data. It uses a specialized method (in _update_attribute) to compute the cdf fast.
        The idea is to subtract values from nodes.attribute in such a way that if the hierarchy is used as an index
        to construct an array of values nodes.attribute, then the cdf is the cumulative sum of that array.
        """
        self._update_attribute()
        cdf_dict = {}
        for level in range(self.get_depth()):
            nodes = self.get_nodes_at_level(level)
            for node in nodes:
                if node._hierarchy is not np.nan:
                    cdf_dict[node._hierarchy - 1] = node.attribute
        cdf = np.zeros(len(cdf_dict))
        for key, value in cdf_dict.items():
            cdf[key] = value
        cdf = np.cumsum(cdf)
        self.cdf = cdf

    def get_quantile(self, quantile) -> int:
        """
        Return the bin that corresponds to the quantile.
        """
        if self.cdf is None:
            self.compute_cdf()
        return np.argmin(np.abs(self.cdf - quantile))

    def set_default(self):
        """
        Set the tree to the default state.
        """
        self.cdf = None
        self._cum_chunks = []
        self._attribute_levels = []

    # ------------------- Inner Functions ------------------- #

    def _get_intervals(self):
        """
        Get the intervals of the b-adic decomposition.
        """
        intervals = []
        depth = self.depth
        for level in range(0, depth):
            intervals_at_level = []
            nodes = self.get_nodes_at_level(level)
            for node in nodes:
                intervals_at_level.append(list(node.data))
            intervals.append(intervals_at_level)
        return intervals

    def _get_nodes_at_level(self, node, level, current_level):
        if current_level == level:
            return [node]

        nodes_at_level = []
        for child in node.children:
            nodes_at_level.extend(self._get_nodes_at_level(child, level, current_level - 1))

        return nodes_at_level

    def _get_range_r(self, node, leaf_data, S, verbose):
        child_nodes = node.children
        for child_node in child_nodes:
            if verbose: print(
                f"\nvisiting node with data {child_node.data} and attribute {child_node.attribute} and S={S}")
            if leaf_data == list(child_node.data) or leaf_data[0] == list(child_node.data)[-1]:
                S += child_node.attribute
                if verbose: print(f"found final node {child_node.data} final S={S}")
                return S
            if leaf_data[0] not in list(child_node.data):
                S += child_node.attribute
                if verbose: print(f"Node not found S={S}")
            if leaf_data[0] in list(child_node.data):
                if verbose: print("\n ------->Going to child node \n")
                S += self._get_range_r(child_node, leaf_data, S=0, verbose=verbose)
                return S

    def _get_depth(self, node):
        max_level = 0
        for child in node.children:
            max_level = max(max_level, self._get_depth(child))
        return max_level + 1

    def _add_hierarchy(self, level):
        branching = self.branching
        nodes = self.get_nodes_at_level(level)
        count_quantum = int(branching ** level)
        count = count_quantum
        if level < self.get_depth() - 1:
            count_skip = 1
            for node in nodes:
                if count_skip != branching:
                    node._hierarchy = count
                    count_skip += 1
                else:
                    count_skip = 1
                count += count_quantum
        else:
            for node in nodes:
                node._hierarchy = count
                count += count_quantum

    def _update_attribute(self):
        """
        Update the attribute of the nodes where the hierarchy is not Nan, in such a way I can construct
        the cdf using the hierarchy as the index and the using the cumulative sum of the attributes.
        """
        depth = self.get_depth()
        for rec_level in range(1, depth):
            for level in range(rec_level, depth):
                cumulative = self._cum_chunks[level - rec_level]
                attributes = self._attribute_levels[level]
                attributes = attributes - cumulative
                # get the indices of Nan
                nan_indices = np.where(np.isnan(attributes))
                # survive only the non Nan values
                self._cum_chunks[level - rec_level] = self._cum_chunks[level - rec_level][nan_indices]
                # update the attribute of the nodes
                self._attribute_levels[level] = attributes
                for node, attribute in zip(self.get_nodes_at_level(level), attributes):
                    node.attribute = attribute
        self.attribute_updated = True


def cumulative_sum_chunks(arr: np.array, chunk_size: int) -> np.array:
    # Calculate the sum over chunks
    return np.add.reduceat(arr, np.arange(0, len(arr), chunk_size))
