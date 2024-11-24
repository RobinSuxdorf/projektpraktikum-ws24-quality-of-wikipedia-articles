import logging
import svm.wp_dump

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    svm.wp_dump.WikipediaDump(
        dump_path="C:/Users/U542596/Desktop/enwiki-20241020-pages-articles-multistream.xml.bz2",
        index_path="C:/Users/U542596/Desktop/enwiki-20241020-pages-articles-multistream-index.txt.bz2",
    ).write_to_csv(
        output_csv_path="C:/Users/U542596/Desktop/enwiki-20241020-pages-articles-filtered.csv",
        num_pages=1000,
    )


if __name__ == "__main__":
    main()
