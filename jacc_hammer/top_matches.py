"""
Enriches `glass_orgs` table with companies house fuzzy matching:

* id_organization (integer) - glass organization reference
* company_number (text) - Best candidate match of company_number
* fuzzy_match_score (integer) - Score from 0 to 100 (100 best, 0 worst) of match quality
    between companies house name corresponding to `company_number_fuzzy`
    and `organization_name` (if a candidate match exists)
"""
import logging
import os
from multiprocessing import Pool

import h5py
import pandas as pd
import psutil
import tqdm
from toolz.curried import curry, do, map, pipe

import mi_scotland
from mi_scotland.utils.pandas import preview

logger = logging.getLogger(__name__)


def _get_top_matches(df):
    """ Get top matches and merge onto names and id's """
    logger.debug("Getting top matches")
    return (
        df.reset_index(drop=True)
        .pipe(lambda x: x.loc[x.groupby("y").sim_mean.idxmax().values])
        .pipe(preview)
    )


def _get_h5_nrows(f, table="matches"):
    nrows = h5py.File(f, "r")[table]["table"].len()
    logger.debug(f"{f} has {nrows} rows")
    return nrows


def get_top_matches(match_fin):
    return pd.read_hdf(match_fin).pipe(_get_top_matches)


@curry
def chunk_row(chunk, tmp_dir="/tmp"):
    """ Get top matches for a dataframe chunk and output interim file """
    chunk_idx, chunk = chunk
    fout = f"{tmp_dir}/top_matches_{chunk_idx}.csv"
    (chunk.pipe(_get_top_matches).to_csv(fout))
    return fout


def get_top_matches_chunked(match_fin, chunksize, stop=None, tmp_dir="/tmp"):
    """ """
    # Make tmp dir if needed
    os.makedirs(tmp_dir, exist_ok=True)

    # Progress bar
    nrows_total = _get_h5_nrows(match_fin)
    nrows = nrows_total if stop is None else stop
    total = pd.np.ceil(nrows / chunksize)
    pbar = curry(tqdm.tqdm, total=total, desc="Chunks reduced")

    n_jobs = psutil.cpu_count(logical=False)
    logger.debug(f"Pool with {n_jobs} jobs")
    with Pool(n_jobs) as p:
        map_ = map
        map_ = curry(p.imap)
        return pipe(
            pd.read_hdf(match_fin, stop=stop, chunksize=chunksize, columns=["x", "y", "sim_mean"]),
            enumerate,
            pbar,
            map_(chunk_row(tmp_dir=tmp_dir)),
            list,  # List of interim filepaths
            map(curry(pd.read_csv, index_col=0)),
            pd.concat,
        ).pipe(_get_top_matches)


if __name__ == "__main__":
    logger = logging.getLogger("mi_scotland")
    debug_file = f"{mi_scotland.project_dir}/debug.log"
    dbg = logging.FileHandler(debug_file)
    dbg.setLevel(logging.DEBUG)
    logger.handlers.append(dbg)
    logger.handlers[0].setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    TEST = True
    TEST_STR = "_TEST" if TEST else ""
    logger.debug(f"TEST: {TEST}")
    if not TEST:
        stop = None
        chunksize = 1e7
    else:
        stop = 100_000
        chunksize = int(5e4)

    project_dir = mi_scotland.project_dir

    # Load glass (y) and ch (x) names
    name_x_fin, name_y_fin = (
        f"{project_dir}/data/interim/companies_house_names{TEST_STR}.csv",
        f"{project_dir}/data/interim/glass_names{TEST_STR}.csv",
    )
    names_y = curry(pd.read_csv, name_y_fin)
    names_x = curry(pd.read_csv, name_x_fin)

    match_fin = f"{project_dir}/data/interim/fuzzy_similarities_glass_ch{TEST_STR}"
    df_top = (
        get_top_matches_chunked(match_fin, chunksize=chunksize, stop=stop)
        .merge(names_y(), left_on="y", right_index=True, validate="1:1")
        .merge(names_x(), left_on="x", right_index=True, validate="m:1")
    )
