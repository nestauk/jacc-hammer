from datasketch import MinHashLSH
import mi_scotland
import logging

logger = logging.getLogger(__name__)


class MinHashLSHHybrid(MinHashLSH):
    def __init__(
        self,
        threshold=0.9,
        num_perm=128,
        weights=(0.5, 0.5),
        params=None,
        storage_config=None,
        prepickle=None,
    ):
        super().__init__(
            threshold, num_perm, weights, params, storage_config, prepickle
        )

        logger.debug(f"{self.b} bands, {self.r} rows")

        # Number of prefix trees
        self.l = self.b
        # Maximum depth of the prefix tree
        self.k = self.r

        # This is the sorted array implementation for the prefix trees
        self.sorted_hashtables = [[] for _ in range(self.l)]

    def index(self):
        """
        Index all the keys added so far and make them searchable.
        """
        for i, hashtable in enumerate(self.hashtables):
            self.sorted_hashtables[i] = [H for H in hashtable.keys()]
            self.sorted_hashtables[i].sort()

    def _query_f(self, minhash, r, b):
        if r > self.k or r <= 0 or b > self.l or b <= 0:
            raise ValueError("parameter outside range")
        # Generate prefixes of concatenated hash values
        hps = [
            self._H(minhash.hashvalues[start : start + r])
            for start, _ in self.hashranges
        ]
        # Set the prefix length for look-ups in the sorted hash values list
        prefix_size = len(hps[0])
        for ht, hp, hashtable in zip(self.sorted_hashtables, hps, self.hashtables):
            i = self._binary_search(len(ht), lambda x: ht[x][:prefix_size] >= hp)
            if i < len(ht) and ht[i][:prefix_size] == hp:
                j = i
                while j < len(ht) and ht[j][:prefix_size] == hp:
                    for key in hashtable[ht[j]]:
                        yield key
                    j += 1

    def query_f(self, minhash, k):
        """
        Return the approximate top-k keys that have the highest
        Jaccard similarities to the query set.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.
            k (int): The maximum number of keys to return.

        Returns:
            `list` of at most k keys.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        results = set()
        r = self.k
        while r > 0:
            for key in self._query_f(minhash, r, self.l):
                results.add(key)
                if len(results) >= k:
                    return list(results)
            r -= 1
        return list(results)

    def _binary_search(self, n, func):
        """
        https://golang.org/src/sort/search.go?s=2247:2287#L49
        """
        i, j = 0, n
        while i < j:
            h = int(i + (j - i) / 2)
            if not func(h):
                i = h + 1
            else:
                j = h
        return i
