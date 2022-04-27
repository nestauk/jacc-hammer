Install: pip install git+https://github.com/nestauk/jacc-hammer.git@legacy

`from jacc_hammer.name_clean import preproc_names` - optional pre-processing function to clean names


```python
from pathlib import Path
from tempfile import TemporaryDirectory

from jacc_hammer.fuzzy_hash import Cos_config, Fuzzy_config, match_names_stream
from pandas import concat, DataFrame

list_of_names_1 = ["alexander",  "jack R", "juan","jack Vines", ]
list_of_names_2 = ["alex",  "jack R", "jack V", ]

# Can set this to an int (0 to 100) and it will ignore matches less than this score
# It's more of an optimisation that matching CB and GtR probably doesn't need 
# so I'd suggest leaving as the default (None)
threshold = None

# `match_names_stream` returns chunks of `chunksize` at once
# to avoid running out of memory.
chunksize = 100_000

# You can also change hyper-parameters used by the two methods in the first
# stage of the matching (identifying possible matches for each name).
# I'd recommend starting with the defaults but if you do need to deviate then
# I've pointed to a few you might want to start with...
#
# ntop - increase this to generate more possible matches (but uses more
#   CPU+memory)
cos_config = Cos_config()
# nperm - higher uses more memory but increases accuracy (don't go less than
#   128)
# threshold - lower yields more possible matches (but uses more CPU+memory).
#   Changing the default in this example to something low yields an extra
#   match.
fuzzy_config = Fuzzy_config()
# fuzzy_config = Fuzzy_config(threshold=0.3)

# `match_names_stream` stores interim results in `tmp_dir`
# to avoid running out of memory
tmp_dir = Path(TemporaryDirectory().name)
tmp_dir.mkdir()

# Do all the matching work!
# `output` is an iterable of "chunks" of `chunksize`
# A "chunk" is a DataFrame like:
#
#    x  y  sim_ratio  sim_jacc  sim_cos  sim_mean
# 0  1  1         75        50       62        62
# 1  1  2         83        60      100        81
# 2  2  1         62        33       62        52
# 3  2  2        100       100      100       100
#
# Where:
# - x is the index of list_of_names_1
# - y is the index of list_of_names_2
# - sim_ratio is the levenshtein similarity (0 to 100)
# - sim_jacc is the (exact) jaccard similarity of the 3-shingles of the names
# is the levenshtein similarity (0 to 100)
# - sim_cos is the cosine similarity of the tfidf entries is the levenshtein
# similarity (0 to 100)
# - sim_mean is the mean of sim_ratio, sim_ratio, and sim_cos is the
# levenshtein similarity (0 to 100)
output = match_names_stream([list_of_names_1, list_of_names_2],
        threshold=threshold, chunksize=chunksize, tmp_dir=tmp_dir,
        cos_config=cos_config, fuzzy_config=fuzzy_config)


def get_top_matches(df: DataFrame) -> DataFrame:
    """For each y get the x and sim_mean corresponding to highest sim_mean."""
    return df.loc[df.groupby("y").sim_mean.idxmax()]

top_matches = get_top_matches(concat(output))

print(top_matches)
# (with Fuzzy config threshold=0.3)
#    x  y  sim_ratio  sim_jacc  sim_cos  sim_mean
# 0  0  0         62        29        0        30
# 1  1  1        100       100      100       100
# 2  1  2         83        60      100        81
# 
# x=0 ("alex"), y=0 ("alexander") has high levenhstein similarity, low jaccard similarity (comparing {"ale", "lex"} and {"ale", "lex", "exa", "xan", "and", "nde", "der"} - 2/7*100 = 29), and zero tfidf cosine similarity (no shared tokens)
# x=1 ("Jack R"), y=1 ("Jack R") perfectly matches on all measures
# x=1 ("Jack R"), y=2 ("Jack V") has high levenshtein similarity, medium-high jaccard similarity (only differing shingle is "k R" vs "k V"), and exact cosine similarity (I think this is because short tokens e.g. the R and V initials are dropped)
```

Example uses...
Createch: https://github.com/nestauk/createch/blob/042731deb658d42f369e3b6da7d0e5803964142a/createch/pipeline/jacchammer/jacchammer.py
Industrial Taxonomy:
https://github.com/nestauk/industrial_taxonomy/blob/dev/industrial_taxonomy/pipeline/glass_house/flow.py (
