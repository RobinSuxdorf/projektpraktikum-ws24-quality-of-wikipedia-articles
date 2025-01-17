import logging
from src.wp import WikipediaDump

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


def main():
    # Download from https://dumps.wikimedia.org/enwiki/:
    # enwiki-yyyymmdd-pages-articles-multistream.xml.bz2 (about 20 GB)
    # enwiki-yyyymmdd-pages-articles-multistream-index.txt.bz2 (about 250 MB)
    
    # Writes about 80 GB of CSV files
    WikipediaDump(
        dump_path="data/wp/enwiki-20241020-pages-articles-multistream.xml.bz2",
        index_path="data/wp/enwiki-20241020-pages-articles-multistream-index.txt.bz2",
    ).write_to_csv(
        good_path="data/wp/good.csv",
        promotional_path="data/wp/promotional.csv",
        neutral_path="data/wp/neutral.csv",
        skipped_path="data/wp/skipped.csv",
        num_pages=-1,
    )


if __name__ == "__main__":
    main()
