import gc
from tempfile import gettempdir
import logging
import os
import uuid
from functools import wraps
from hashlib import sha1
from itertools import chain, repeat, starmap
from multiprocessing import Pool
from pathlib import Path
from typing import Any, BinaryIO, Callable, NamedTuple, Optional, Tuple
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as ss
import tqdm
from cytoolz.curried import (compose, curry, do, filter, get, juxt, map,
                             partition_all, peek, pipe)
from datasketch import LeanMinHash, MinHash
from fuzzywuzzy import process
from Levenshtein import ratio
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import im_minerva
from im_minerva.fuzzy_hash import MinHashLSHHybrid

NUM_THREADS = "4"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
TMP_DIR = gettempdir()


logger = logging.getLogger(__name__)


@curry
def vectorise_names(names, vectoriser, tfidf_transformer):
    """ Vectorise the input names (jointly) and transform to tf-idf (independently)

    Args:
        names ((list of str, list of str)): Two lists of strings to vectorise
        vectoriser (class): Class to vectorise `names` with.
            Must implement a scikit-learn style fit, transform API.
        tfidf_transformer (class): Class to transform `names` with.
            Must implement a scikit-learn style fit, transform API.

    Returns:
       (ss.sparse.csr.csr_matrix, ss.sparse.csr.csr_matrix)
    """

    return pipe(
        names,
        do(lambda x: vectoriser.fit(x[0] + x[1])),  # Jointly fit vectoriser
        # Vectorise and Tf-idf transform independently
        map(compose(tfidf_transformer.fit_transform, vectoriser.transform)),
        tuple,
    )


def _get_csr_ntop_idx_data(csr_row, ntop=10):
    """ Get the `ntop` indices with the highest cosine similarity"""
    nnz = csr_row.getnnz()
    if nnz <= ntop:
        return csr_row.indices
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        return csr_row.indices[arg_idx]


@curry
def _cossim_top(A, B, ntop=10):
    """ Calculate the cosine similarity of A and B, return top `ntop` vals"""
    return [_get_csr_ntop_idx_data(row, ntop) for row in A.dot(B)]


@curry
def _sparse_chunks(X, chunksize, axis=0):
    """ Split `X` into chunks of `chunksize` along `axis` w/ progress bar """
    return [X[i : i + chunksize] for i in range(0, X.shape[axis], chunksize)]


@curry
def top_cosine_similarities(
    Xs: tuple,
    chunksize: Optional[int] = None,
    ntop: Optional[int] = 10,
    map_: Callable = map,
) -> iter:
    """ Top `ntop` most cossine similar entries of `Xs[1]` to each `Xs[0]`

    Calculated in chunks of `chunksize` for memory efficiency.

    Args:
        Xs (Two-tuple of sparse matries): TF-idf matrices
        chunksize (int, optional): Number of rows of `Xs[0]` to dot product
            with `Xs[1]` at once.
        ntop (int, optional): Number of approximate nearest neighbours to return.

    Returns:
       - iter of tuples (row, col, similarity)
       - `None` if all cosine similarities are zero
    """
    X_a, X_b = Xs

    if (chunksize is None) or (chunksize > X_a.shape[0]):
        _chunksize = X_a.shape[0]
    else:
        _chunksize = chunksize
    nchunks = np.ceil(X_a.shape[0] / _chunksize).astype(int)

    # Get cosine similarities in chunks
    return pipe(
        X_a,
        _sparse_chunks(chunksize=_chunksize),
        map_(_cossim_top(B=X_b.T, ntop=ntop)),
        curry(tqdm.tqdm, desc="chunked dot product", total=nchunks),
        chain.from_iterable,
    )


@curry
def _lok_to_csr(X, shape):
    """ List of keys to sparse binary CSR matrix

    Args:
        X (list of array-like): The i-th element of `X` is a list indicating
            the non-zero columns corresponding to the i-th row.
        shape ((int, int)): Shape of output sparse matrix.

    Returns:
        scipy.sparse.csr.csr_matrix
    """

    if len(X) != shape[0]:
        raise ValueError(
            f"Length of `X` ({len(X)}) does not match `shape[0]` ({shape[0]})"
        )

    indptr = [0]
    indices = []
    counter = 0
    for x in X:
        x_ = sorted(x)
        indices += x_
        counter += len(x_)
        indptr.append(counter)

    return ss.csr_matrix(
        (np.ones(len(indices), dtype=np.bool), indices, indptr), shape=shape
    )


@curry
def _lok_to_coo(X, shape):
    """ List of keys to sparse binary COO matrix

    Args:
        X (list of array-like): The i-th element of `X` is a list indicating
            the non-zero columns corresponding to the i-th row.
        shape ((int, int)): Shape of output sparse matrix.

    Returns:
        scipy.sparse.coo.coo_matrix
    """
    if len(X) != shape[0]:
        raise ValueError(
            f"Length of `X` ({len(X)}) does not match `shape[0]` ({shape[0]})"
        )
    i = []
    for idx, x in enumerate(X):
        i.extend(repeat(idx, len(x)))
    j = list(chain.from_iterable(X))

    return ss.coo_matrix((np.ones(len(i), dtype=np.bool), (i, j)), shape=shape)


@curry
def nearest_neighbour_matrix(results, shape, binary=True):
    """ Convert (row, col, similarity) to CSR sparse matrix

    Args:
        results (iterable): May be None or a tuple of (row, col, similarity)
        shape ((int, int)): Shape of sparse matrix - each dimension should be
            the number of entries in the datasets being compared.
        binary (bool, optional): If True, binarise the similarities.

    Returns:
       scipy.sparse.csr.csr_matrix
    """
    dtype = np.bool if binary else None

    return pipe(
        results,
        filter(lambda x: x is not None),
        chain.from_iterable,
        list,
        np.array,
        curry(np.reshape, newshape=(-1, 3)),  # Deals with empty case
        lambda x: ss.coo_matrix(
            (x[:, 2], (x[:, 0], x[:, 1])), shape=shape, dtype=dtype
        ).tocsr(),
    )


class Cos_config(NamedTuple):
    ntop: int = 10
    chunksize: int = 1_000


@curry
def main_cos(
    names: list,
    fout_Xs: str,
    map_: Callable = map,
    config: Cos_config = Cos_config(),
    vec_kws: Optional[dict] = None,
    trans_kws: Optional[dict] = None,
) -> iter:
    """ """
    logger.debug(config)

    if vec_kws is None:
        vec_kws = {}
    if trans_kws is None:
        trans_kws = {}

    vec = CountVectorizer(**vec_kws)
    tf = TfidfTransformer(**trans_kws)

    X = vectorise_names(names, vectoriser=vec, tfidf_transformer=tf)
    joblib.dump(X, fout_Xs)  # Cache for use as similarity scorer
    return top_cosine_similarities(
        X, ntop=config.ntop, chunksize=config.chunksize, map_=map_
    )


@curry
def construct_minhashes(names, minhash_, map_=map):
    """ Construct MinHashes of `names`

    Args:
        names (list of str): Names to minhash
        minhash_ (function): Funtion to minhash `names` with
        map_ (function, optional): map function to use

    Returns:
        dict with keys enumerating values that are MinHash-like
    """

    return dict(enumerate(map_(minhash_, tqdm.tqdm(names, desc="Construct minhashes"))))


@curry
def lsh_insert(minhashes, lsh, lowmem=False, cutoff=0):
    """ Construct a Locality Sensitive Hashing (LSH) for Jaccard threshold & top-k query

    Uses `datasketch.MinHashLSHHybrid` such that if a query returns too
    many results then a given number of approximate nearest neighbours
    can be returned using the same hashtable.

    Args:
        minhashes (list of MinHash-like): Minhashes to insert into `lsh`
        lsh (MinHashLSHHybrid): The LSH data structure to insert minhashes into
        lowmem (bool, optional): If True, consumes `minhashes` when inserted.
        cutoff (int, optional): Zero cutoff does not truncate results.
            Positive uses LSHForest methods to find `cutoff` nearest neighbours.

    Returns:
        MinHashLSHHybrid
    """

    desc = "Inserting minhashes"
    with lsh.insertion_session() as session:
        if lowmem:
            k = list(minhashes.keys())
            for i in tqdm.tqdm(k, desc=desc):
                session.insert(i, minhashes.pop(i))
        else:
            for i, minhash in tqdm.tqdm(minhashes.items(), desc=desc):
                session.insert(i, minhash)

    if cutoff > 0:
        # Creates sorted hashtables (Extra mem cost)
        logger.debug("Indexing LSH")
        lsh.index()
        logger.debug("Indexed LSH")

    return lsh


@curry
def _query_idx(lsh, name, minhash_, cutoff):
    """ Query the LSH for `name`, returning indices.

    If the initial query returns more than `cutoff` candidates then
    the `cutoff` approximate nearest neighbours are returned.

    Args:
        lsh (MinHashLSHHybrid): The LSH to query
        name (str): Query string.
        minhash_ (function): The function to minhash `name` with
        cutoff (int, optional): Number of query results above which to
            return `cutoff` approximate nearest neighbours.

    Returns:
        list of integers: Indices corresponding to position of matches
            in `self.names_b`.
    """

    mh = minhash_(name)
    results = lsh.query(mh)
    if len(results) > cutoff:
        results = lsh.query_f(mh, cutoff)
        # self._high_recall.append(name)
    # if len(results) == 0:
    # self._zero_recall.append(name)

    return np.array(results, dtype=np.uint32)


class Fuzzy_config(NamedTuple):
    hashobj: Callable = sha1
    k: int = 3
    n_bytes: int = 4
    num_perm: int = 128
    threshold: Optional[float] = 0.5
    weights: Tuple[float, float] = (0.5, 0.5)
    storage_config: Any = None
    cutoff: int = 1000
    lowmem: bool = True


@curry
def main_fuzzy(names, map_, config=Fuzzy_config()):
    """ """
    logger.debug(config)
    minhash_ = lean_minhash(
        num_perm=config.num_perm,
        hashobj=config.hashobj,
        n_bytes=config.n_bytes,
        k=config.k,
    )

    lsh = MinHashLSHHybrid(
        threshold=config.threshold,
        num_perm=config.num_perm,
        storage_config=config.storage_config,
        weights=config.weights,
    )

    # Build LSH Query
    query = pipe(
        names[1],
        construct_minhashes(minhash_=minhash_, map_=map_),
        lsh_insert(lsh=lsh, cutoff=config.cutoff, lowmem=config.lowmem),
        _query_idx(minhash_=minhash_, cutoff=config.cutoff),
    )

    return pipe(
        names[0], map(query), curry(tqdm.tqdm, total=len(names[0]), desc="LSH Queries"),
    )


@curry
def minhash(name, num_perm, hashobj, n_bytes, k):
    """ Shingle and MinHash a string.

    Args:
        name (str): Input string.
        num_perm (int): Number of minhash permutations.
        hashobj (function): Hash function.
        n_bytes (int): Size in bytes of each minhash.
        k (int): Shingle size.

    Returns:
        datasketch.MinHash
    """
    minhash = MinHash(num_perm=num_perm, hashobj=hashobj, n_bytes=n_bytes)
    for d in ngrams(name, k):
        minhash.update("".join(d).encode("utf-8"))

    return minhash


@curry
def lean_minhash(name, num_perm, hashobj, n_bytes, k):
    """ LeanMinHash `minhash()`

    Args:
        name (str): Input string.
        num_perm (int): Number of minhash permutations.
        hashobj (function): Hash function.
        n_bytes (int): Size in bytes of each minhash.
        k (int): Shingle size.

    Returns:
        datasketch.LeanMinHash
    """
    return LeanMinHash(minhash(name, num_perm, hashobj, n_bytes, k))


def lsh_query(name, lsh, minhash_, cutoff):
    """ Query the LSH for `name`, return names and similarities.

    If the initial query returns more than `cutoff` candidates then
    the `cutoff` approximate nearest neighbours are returned. If `self.lsh`
    is not an instance of MinHashLSHHybrid and `self.lsh_f` does not exist
    this returns an error.

    Args:
        name (str): Query string.
        cutoff (int, optional): Number of query results above which to
            return `cutoff` approximate nearest neighbours.

    Returns:
        list of tuples: Each tuple contains a name similarity pair.

            [('Acme Corp', 100),
            ('Ace Co', 79)]
    """
    results = _query_idx(lsh, name, minhash_, cutoff)
    return process.extract(name, results)


def mh_jacc(query, choice, query_idx, choice_idx):
    """ MinHash and calculate jaccard similarity of a
    query string and a list of candidates.

    Args:

    Returns:
        list[int]: list of Jaccard similarities
            between 0 and 100.
    """
    return [np.round(jaccard(query, c) * 100) for c in choice]


def jaccard(x, y):
    x_shingle = set(ngrams(x, n=3))
    y_shingle = set(ngrams(y, n=3))
    try:
        return len(x_shingle & y_shingle) / len(x_shingle | y_shingle)
    except ZeroDivisionError:
        return 0


# TODO: Handle empty strings and strings of len < `n` => zero division error


def cos_scorer_gen(Xs):
    def cos_scorer(query, choice, query_idx, choice_idx):
        # Cosine
        return np.round(
            Xs[0][query_idx].dot(Xs[1][choice_idx].T).toarray().ravel() * 100
        )

    return cos_scorer


def cos_scorer(Xs, query, choice, query_idx, choice_idx):
    # Cosine
    return np.round(Xs[0][query_idx].dot(Xs[1][choice_idx].T).toarray().ravel() * 100)


##
def ratio_wrap(query, choice, query_idx, choice_idx, *args, **kwargs):
    return [int(np.round(ratio(query, c) * 100)) for c in choice]


@curry
def _get_chunk_similarity(scorers, names, result_chunk, progress=True):
    """
    Args:
        scorers (dict): Keys are scorer names, values are scorer functions
        names ((list of str, list of str)): Names to calculate similarities of
        result_chunk (dict): (query index, choice index list) pairs
        progress (bool, optional): If True use progress counter

    Returns:
        numpy.array, numpy.array
    """
    m = len(scorers.keys())
    n = sum(map(len, result_chunk.values()))
    output_idx = np.empty((n, 2), dtype=np.uint32)
    output_sim = np.empty((n, m), dtype=np.uint8)
    logger.debug(f"allocated matrix size: {output_sim.shape}")

    i = 0
    for query_idx in tqdm.tqdm(list(result_chunk.keys()), disable=not progress):
        choice_idxs = list(result_chunk.pop(query_idx))
        q = names[0][query_idx]
        step = len(choice_idxs)
        output_idx[i : i + step, 0] = query_idx
        output_idx[i : i + step, 1] = choice_idxs
        output_sim[i : i + step, :] = scorer(
            scorers, q, [names[1][cand] for cand in choice_idxs], query_idx, choice_idxs
        )
        i += step
    return output_idx, output_sim


def scorer(scorers, query, choice, query_idx, choice_idx):
    """ Apply `scorers` """
    return np.array(
        # TODO: scorers should only take query and choice
        [s(query, choice, query_idx, choice_idx) for s in scorers.values()]
    ).T


def near_neighbour_similarities(names, scorers, X, chunksize=None, map_=map):
    """

    Args:
       names ():
       X ():

    """

    if chunksize is None:
        _get_chunk_similarity_ = _get_chunk_similarity(scorers, names, progress=True)
        chunksize = len(names[0])
        nchunks = 1
    else:
        num_k = len(names[0])
        nchunks = np.ceil(num_k / chunksize).astype(int)
        logger.debug(f"chunk info: {num_k} {chunksize} {nchunks}")
        _get_chunk_similarity_ = _get_chunk_similarity(scorers, names, progress=False)
    filter_empty = filter(compose(len, get(1)))

    return pipe(
        X,
        enumerate,
        # TODO: filter empty here?:
        partition_all(chunksize),
        map_(compose(_get_chunk_similarity_, dict, filter_empty)),
        curry(tqdm.tqdm, total=nchunks, desc="Chunks"),
    )


def match_chunk_to_df(
    chunk: tuple, scorer_names: list, threshold: float = None
) -> pd.DataFrame:
    output_idx, output_sim = chunk
    return pd.DataFrame(
        np.hstack(
            [output_idx, output_sim, output_sim.mean(1).astype(np.uint8)[:, np.newaxis]]
        ),
        columns=["x", "y", *scorer_names, "sim_mean"],
    ).pipe(lambda x: x if threshold is None else x.query(f"sim_mean > {threshold}"))


def stream_sim_chunks_to_hdf(out: iter, fout: str) -> str:
    """ Stream an iterable of dataframes to hdf """
    # Initialise HDF dataframe
    head, out = peek(out)
    columns = head.columns
    logger.debug(f"Initialising file: {fout}")
    (
        pd.DataFrame([], columns=columns).to_hdf(
            fout, format="table", append=True, mode="w", key="matches"
        )
    )

    # Output chunk
    list(
        chunk.pipe(do(lambda x: logger.debug(f"saving {x.shape[0]}"))).to_hdf(
            fout,
            format="table",
            append=True,
            mode="a",
            key="matches",
            data_columns=["x", "y", "sim_mean"],
        )
        for chunk in out
    )
    return fout


def np_buff_read(fin: BinaryIO, fin_idx: str) -> np.ndarray:
    return map(
        lambda x: np.frombuffer(fin.read(4 * x), dtype=np.uint32),
        np.fromfile(fin_idx, dtype=np.uint32),
    )


def np_buff_write(fout: BinaryIO, fout_idx: BinaryIO, x: np.ndarray) -> np.ndarray:
    """
    fout stores array as bytes
    fout_idx stores length of each iterable
    x array
    """
    fout.write(x.astype(np.uint32).tobytes())
    fout_idx.write(np.uint32(x.size))
    return x


def np_buffer(fun: Callable, path: str) -> Callable:
    """ Cached numpy buffer stream """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if not os.path.exists(path):
            nn = fun(*args, **kwargs)

            with open(path, "wb") as fout:
                with open(path + "_idx", "wb") as fout_idx:
                    [np_buff_write(fout, fout_idx, x) for x in nn]
            del nn
            gc.collect()
        fin = open(path, "rb")  # Read here keeps file open during call of `fun`
        return np_buff_read(fin, path + "_idx")

    return wrapper


def match_names(
    names: list,
    tmp_dir: str = TMP_DIR,
    chunksize: int = 10000,
    threshold: int = None,
    cos_config: Cos_config = Cos_config(),
    fuzzy_config: Fuzzy_config = Fuzzy_config(),
) -> pd.DataFrame:
    """ """

    fout_Xs = f"{tmp_dir}/cos_tfidf_cache"

    n_jobs = psutil.cpu_count(logical=False)
    with Pool(n_jobs) as p:
        map_ = curry(p.imap)
        X = pipe(
            names,
            juxt(
                curry(main_cos, map_=map_, fout_Xs=fout_Xs, config=cos_config),
                curry(main_fuzzy, map_=map_, config=fuzzy_config),
            ),
            lambda x: zip(x[0], x[1]),
            curry(starmap, np.union1d),
        )

    cos_scorer_ = curry(cos_scorer, joblib.load(fout_Xs))
    scorers = {"sim_ratio": ratio_wrap, "sim_jacc": mh_jacc, "sim_cos": cos_scorer_}

    return pipe(
        X,
        curry(
            near_neighbour_similarities, names, scorers, chunksize=chunksize, map_=map
        ),
        map(
            curry(
                match_chunk_to_df, scorer_names=[*scorers.keys()], threshold=threshold
            )
        ),
        pd.concat,
    )


def match_names_stream(
    names: list,
    tmp_dir: str = TMP_DIR,
    chunksize: int = 10000,
    threshold: int = None,
    cos_config: Cos_config = Cos_config(),
    fuzzy_config: Fuzzy_config = Fuzzy_config(),
) -> iter:
    """ """

    names = list(map(list, names))
    # Automatically deal with stream to/from buffer
    main_cos_ = np_buffer(main_cos, f"{tmp_dir}/cos_stream")
    main_fuzzy_ = np_buffer(main_fuzzy, f"{tmp_dir}/fuzzy_stream")

    fout_Xs = f"{tmp_dir}/cos_tfidf_cache"

    n_jobs = psutil.cpu_count(logical=False)
    with Pool(n_jobs) as p:
        map_ = curry(p.imap)

        X = pipe(
            names,
            juxt(
                curry(main_cos_, map_=map_, fout_Xs=fout_Xs, config=cos_config),
                curry(main_fuzzy_, map_=map_, config=fuzzy_config),
            ),
            lambda x: zip(x[0], x[1]),
            curry(starmap, np.union1d),
        )

        cos_scorer_ = curry(cos_scorer, joblib.load(fout_Xs))
        scorers = {"sim_ratio": ratio_wrap, "sim_jacc": mh_jacc, "sim_cos": cos_scorer_}

    return pipe(
        X,
        curry(
            near_neighbour_similarities, names, scorers, chunksize=chunksize, map_=map
        ),
        map(
            curry(
                match_chunk_to_df, scorer_names=[*scorers.keys()], threshold=threshold
            )
        ),
    )
