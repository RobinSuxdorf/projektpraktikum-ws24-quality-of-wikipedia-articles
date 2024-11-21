import logging
import svm.dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    svm.dataset.convert_wp_dump(
        dump_path="C:/Users/U542596/Desktop/enwiki-20241020-pages-articles-multistream.xml.bz2",
        index_path="C:/Users/U542596/Desktop/enwiki-20241020-pages-articles-multistream-index.txt.bz2",
        output_csv_path="C:/Users/U542596/Desktop/enwiki-20241020-pages-articles-filtered.csv",
        num_pages=-1,
    )


if __name__ == "__main__":
    main()
