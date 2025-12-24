"""
Utility data structures for operator solvers.
"""
from typing import Callable, Union, Sequence, List, Tuple
import math
import bisect


class TreeSegment:
    """
    A simple and index-based implementation of a lazy segment tree
    for range updates and queries.
    """
    def __init__(
        self,
        data: Sequence[Union[int, float]],
        op: Callable[[Union[int, float], Union[int, float]], Union[int, float]],
        identity: Union[int, float],
        sum_like: bool = True,
        add_neutral: Union[int, float] = 0,
    ):

        self.num_elements = len(data)
        self.op = op
        self.identity = identity

        # If True, node aggregate scales with segment length on range add (sum queries).
        # If False, node aggregate shifts uniformly (min/max queries).
        self.sum_like = sum_like

        # Find the number of leaves of the complete binary tree
        self.num_leaves = 1
        while self.num_leaves < self.num_elements:
            self.num_leaves *= 2

        # Initialize the data and lazy arrays
        self.data = [self.identity] * (2 * self.num_leaves)
        # Lazy stores pending additive increments to elements; neutral is 0, not `identity`.
        self.lazy = [add_neutral] * (2 * self.num_leaves)

        # Build the tree
        for i in range(self.num_elements):
            self.data[self.num_leaves + i] = data[i]
        for i in range(self.num_leaves - 1, 0, -1):
            self.data[i] = self.op(self.data[2 * i], self.data[2 * i + 1])


    # ================ Inner methods ===============
    # Apply the increment update to node x
    def _apply(self, x: int, val: Union[int, float], length: int):
        if self.sum_like:
            # Sum aggregator: whole segment increases by val per element
            self.data[x] += val * length
        else:
            # Min/Max aggregator: aggregate shifts uniformly by val
            self.data[x] += val
        if x < self.num_leaves:
            self.lazy[x] += val


    # Push the pending updates from node x to its children
    def _push(self, x: int, length: int):
        if self.lazy[x] != 0:
            self._apply(2 * x, self.lazy[x], length // 2)
            self._apply(2 * x + 1, self.lazy[x], length // 2)
            self.lazy[x] = 0


    # Update the range [l, r) by adding val
    def _update(self, l: int, r: int, val: Union[int, float], x: int, lx: int, rx: int):
        # No overlap
        if lx >= r or rx <= l:
            return

        # Total overlap
        if lx >= l and rx <= r:
            self._apply(x, val, rx - lx)
            return
        self._push(x, rx - lx)

        # Partial overlap
        m = (lx + rx) // 2
        self._update(l, r, val, 2 * x, lx, m)
        self._update(l, r, val, 2 * x + 1, m, rx)
        self.data[x] = self.op(self.data[2 * x], self.data[2 * x + 1])


    # Query the aggregate value over interval [l, r)
    def _query(self, l: int, r: int, x: int, lx: int, rx: int) -> Union[int, float]:
        # No overlap
        if lx >= r or rx <= l:
            return self.identity

        # Total overlap
        if lx >= l and rx <= r:
            return self.data[x]

        # Partial overlap
        self._push(x, rx - lx)
        m = (lx + rx) // 2
        left = self._query(l, r, 2 * x, lx, m)
        right = self._query(l, r, 2 * x + 1, m, rx)
        return self.op(left, right)


    # =============== Public methods ===============
    # Update the range [l, r) by adding val
    def update(self, l: int, r: int, val: Union[int, float]):
        """
        Update the range [l, r) by adding val.
        """
        self._update(l, r, val, 1, 0, self.num_leaves)


    # Query the aggregate value over interval [l, r)
    def query(self, l: int, r: int) -> Union[int, float]:
        """
        Query the aggregate value over interval [l, r).
        """
        return self._query(l, r, 1, 0, self.num_leaves)



class MinMaxPfsumArray:
    """
    Square root decomposition structure to maintain an array
    supporting insertions/deletions and range min/max prefix sum queries.
    """
    class Block:
        """
        A block data structure within the array.
        This maintains block statistics and managing point insert/delete
        for efficient prefix queries.
        """
        def __init__(self, data):
            self.arr = data[:]          # list of data
            self.size = len(self.arr)   # number of elements
            self.recalc()


        def recalc(self):
            """
            Recompute size, sum, min_pref, max_pref.
            This is done in an exhaustive manner, as this is still O(sqrt(n)) overall.
            """
            self.size = len(self.arr)
            self.sum = sum(self.arr)
            csum = 0
            mn = 10**18
            mx = -10**18
            for x in self.arr:
                csum += x
                mn = min(mn, csum)
                mx = max(mx, csum)
            self.min_pref = mn
            self.max_pref = mx


        def insert(self, idx, entry):
            """
            Insert entry at position idx in this block.
            """
            self.arr.insert(idx, entry)
            self.recalc()


        def erase(self, idx):
            """
            Remove entry at position idx in this block.
            """
            del self.arr[idx]
            self.recalc()



    def __init__(self, data: List[int]):
        """
        Initialize the block array from the given array.
        This includes building the block structure and a block indexing:
        """
        assert data
        self.block_arr = []
        self.num_data = 0                      # total number of elements
        self.block_prefix: List[int] = []       # prefix element count (len = #blocks)

        self.build(data)


    def build(self, data: List[int]):
        """
        Build the block array from the given array.
        """
        self.block_arr.clear()

        self.num_data: int = len(data)
        self.block_size = max(0, int(math.sqrt(self.num_data))) + 2

        for i in range(0, self.num_data, self.block_size):    # note the step size
            self.block_arr.append(self.Block(data[i:i + self.block_size]))
        self.num_block = len(self.block_arr)

        self._rebuild_indexing()


    def _rebuild_indexing(self):
        """
        Recompute block_prefix in O(#blocks)
        """
        self.block_prefix = []

        cumid = 0
        for b in self.block_arr:
            self.block_prefix.append(cumid)
            cumid += b.size

        self.num_data = cumid


    def _find_block(self, idx: int) -> Tuple[int, int]:
        """
        Find the block containing the given global idx.
        Returns (block_index, inner_index).
        """
        assert self.block_arr, "No blocks present"
        assert 0 <= idx < self.num_data, "Index out of bounds"
        if idx > self.num_data:
            idx = self.num_data

        # Retrieve bid as largest prefix <= idx
        # bid is the block index
        bid = bisect.bisect_right(self.block_prefix, idx) - 1
        # Retrieve iid as inner index within block
        # iid is the index within block bid
        iid = idx - self.block_prefix[bid]

        return bid, iid


    def insert(self, idx, val):
        """
        Insert val at position idx in the overall array.
        Return None.
        """
        # Support append at end without relying on _find_block assertion
        if idx == self.num_data:
            if not self.block_arr:
                self.block_arr.append(self.Block([val]))
            else:
                last = self.block_arr[-1]
                # If last block is too large, start a new block to keep sqrt decomposition
                if last.size >= 2 * self.block_size:
                    self.block_arr.append(self.Block([val]))
                else:
                    last.insert(last.size, val)

            # Update data structure
            self.num_data += 1
            self._rebuild_indexing()
            return

        # Retrieve block and inner indices, then perform insertion
        bid, iid = self._find_block(idx)
        blk = self.block_arr[bid]
        blk.insert(iid, val)

        # If block grows too large, split it to maintain O(sqrt n) bounds
        if blk.size > 2 * self.block_size:
            arr = blk.arr
            mid = len(arr) // 2
            left = self.Block(arr[:mid])
            right = self.Block(arr[mid:])
            self.block_arr[bid:bid + 1] = [left, right]

        # Update data structure
        self.num_data += 1
        self._rebuild_indexing()


    def delete(self, idx):
        """
        Remove element at position idx in the overall array.
        Return None.
        """
        # Retrieve block and inner indices, then perform deletion
        bid, iid = self._find_block(idx)
        self.block_arr[bid].erase(iid)

        # If block becomes empty, drop it; otherwise consider merging small blocks
        if self.block_arr[bid].size == 0:
            del self.block_arr[bid]
        else:
            # Merge with next block if both are small enough
            min_size = max(1, self.block_size // 2)
            if self.block_arr[bid].size < min_size:
                # Prefer merge with next if possible
                if bid + 1 < len(self.block_arr):
                    nxt = self.block_arr[bid + 1]
                    if self.block_arr[bid].size + nxt.size <= 2 * self.block_size:
                        merged = self.block_arr[bid].arr + nxt.arr
                        self.block_arr[bid:bid + 2] = [self.Block(merged)]

                # Otherwise try merge with previous
                elif bid - 1 >= 0:
                    prv = self.block_arr[bid - 1]
                    if prv.size + self.block_arr[bid].size <= 2 * self.block_size:
                        merged = prv.arr + self.block_arr[bid].arr
                        self.block_arr[bid - 1:bid + 1] = [self.Block(merged)]

        # Update data structure
        self.num_data -= 1
        self._rebuild_indexing()


    def query_min_prefix(self, l, r):
        """
        Query the minimum GLOBAL prefix sum value attained at any position k in [l, r-1].
        (Global means we do NOT subtract the prefix up to l-1; i.e. we look at the array's
        cumulative sum up to k.)
        """
        ans = 10**18
        pos = 0
        prefix = 0          # global prefix up to current processed position
        ans = 10**18        # track minimum global prefix encountered inside [l,r)
        pos = 0             # starting index of current block
        for b in self.block_arr:
            blen = b.size
            # Entire block lies before l: add its sum and skip
            if pos + blen <= l:
                prefix += b.sum
                pos += blen
                continue
            if pos >= r:
                break

            start = max(0, l - pos)      # first index inside block to include
            end   = min(blen, r - pos)    # exclusive end index inside block

            # If we start mid-block, advance prefix by skipped elements
            if start > 0:
                for i in range(start):
                    prefix += b.arr[i]

            if start == 0 and end == blen:
                # Full block contained in range: min within block is prefix + b.min_pref
                ans = min(ans, prefix + b.min_pref)
                prefix += b.sum
            else:
                # Partial block: iterate needed portion
                for i in range(start, end):
                    prefix += b.arr[i]
                    ans = min(ans, prefix)

            pos += blen

        # If interval empty, return +inf (caller responsibility). Could also return None.
        return ans


    def query_max_prefix(self, l, r):
        """
        Query the maximum GLOBAL prefix sum value attained at any position k in [l, r-1].
        (Global means we do NOT subtract prefix up to l-1.)
        """
        ans = float('-inf')
        pos = 0
        prefix = 0
        ans = float('-inf')
        pos = 0
        for b in self.block_arr:
            blen = b.size
            if pos + blen <= l:
                prefix += b.sum
                pos += blen
                continue
            if pos >= r:
                break

            start = max(0, l - pos)
            end   = min(blen, r - pos)

            if start > 0:
                for i in range(start):
                    prefix += b.arr[i]

            if start == 0 and end == blen:
                ans = max(ans, prefix + b.max_pref)
                prefix += b.sum
            else:
                for i in range(start, end):
                    prefix += b.arr[i]
                    ans = max(ans, prefix)

            pos += blen

        return ans


    def get_data_point(self, idx) -> int:
        """
        Retrieve the value at global index idx.
        Raises IndexError if idx is out of bounds.
        """
        if idx < 0 or idx >= self.num_data:
            raise IndexError("Index out of bounds")

        bid, iid = self._find_block(idx)

        return self.block_arr[bid].arr[iid]


    def get_data_segment(self, l: int, r: int) -> List[int]:
        """
        Retrieve a contiguous data segment for half-open interval [l, r).
        Raises IndexError if the range is invalid or out of bounds.
        """
        if l < 0 or r < 0 or l > r or r > self.num_data:
            raise IndexError("Invalid segment range")

        result: List[int] = []
        pos = 0
        for b in self.block_arr:
            blen = b.size
            if pos >= r:
                break
            if pos + blen <= l:
                pos += blen
                continue

            start = max(0, l - pos)
            end = min(blen, r - pos)
            if end > start:
                result.extend(b.arr[start:end])

            pos += blen

        return result


    def get_data(self) -> List[int]:
        """
        Retrieve the entire data array
        """
        return self.get_data_segment(0, self.num_data)


if __name__ == "__main__":
    datalist = [3, -2, 5, -1, 4, -3, 2]
    segment_manager = MinMaxPfsumArray(datalist)
    print("Initial segment array:", datalist)
    print("Min prefix sum [2, 5):", segment_manager.query_min_prefix(2, 5))
    print("Max prefix sum [0, 7):", segment_manager.query_max_prefix(0, 7))

    segment_manager.insert(3, -4)
    print("After inserting -4 at index 3:")
    print("Segment array:", segment_manager.get_data())
    print("Min prefix sum [2, 5):", segment_manager.query_min_prefix(2, 5))
    print("Max prefix sum [0, 7):", segment_manager.query_max_prefix(0, 7))

    segment_manager.delete(5)
    print("After deleting element at index 5:")
    print("Segment array:", segment_manager.get_data())
    print("Min prefix sum [2, 5):", segment_manager.query_min_prefix(2, 5))
    print("Max prefix sum [0, 7):", segment_manager.query_max_prefix(0, 7))
