def replace_synonyms(x, synonyms):
    """
    """

    for synonym, common_name in synonyms.items():
        x = x.str.replace(synonym, common_name)

    return x

def preproc_names(x, stopwords=None, synonyms=None):
    if stopwords is None:
        stopwords = []  # ["ltd", "llp", "limited"]
    if synonyms is None:
        synonyms = {
            'public limited company': 'plc',
            'limited company': 'ltd',
            'limited': 'ltd',
        }
    return (x
            .astype(str)
            .str.strip()
            .str.replace(r"([a-z])([A-Z][a-z])",
                         lambda x: x[1] + '_' + x[2].lower())
            .str.lower()
            .str.replace(r".*www\.(.*?)\..*", r"\1")
            .str.replace(r"https?://(.*)\..*", r"\1")
            .str.replace(r"\.com|\.co\.uk|\.ac\.uk|\.org\.uk", "")
            .pipe(replace_synonyms, synonyms)
            .str.replace(f"{'|'.join(stopwords)}", "")
            .str.replace('\.|\s+', ' ')
            .str.replace("[^a-z0-9 ]", '')  # Alphanum only
            .str.strip()  # Remove added spaces
           )

if __name__ == "__main__":

    glass = (glass
             .assign(org_name_clean=lambda x: x.org_name.pipe(preproc_names))
             .query("org_name_clean != ''")
             .pipe(preview)
             )

    ch = (ch.assign(company_name_clean=lambda x: x.company_name.pipe(preproc_names))
          .query("company_name_clean != ''")
          .pipe(preview)
          )

    gtr = (gtr.assign(name_clean=lambda x: x.name.pipe(preproc_names))
          .query("name_clean != ''")
          .pipe(preview)
          )
