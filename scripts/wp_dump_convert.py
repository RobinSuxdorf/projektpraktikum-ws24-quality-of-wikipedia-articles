"""Convert a Wikipedia XML dump into CSV files

Author: Johannes Krämer
"""

import logging
import sys
from src.wp import WikipediaDump
from src import (
    get_argument_parser,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    # Download from https://dumps.wikimedia.org/enwiki/:
    # enwiki-yyyymmdd-pages-articles-multistream.xml.bz2 (about 20 GB)
    # enwiki-yyyymmdd-pages-articles-multistream-index.txt.bz2 (about 250 MB)

    # Writes about 80 GB of CSV files

    parser = get_argument_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        convert_dump(config)
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.info("Exiting with return code 1")
        sys.exit(1)


def convert_dump(config: dict) -> None:
    input_config = config.get("input")
    output_config = config.get("output")
    WikipediaDump(
        dump_path=input_config.get("dump"),
        index_path=input_config.get("index"),
    ).write_to_csv(
        good_path=output_config.get("good"),
        promotional_path=output_config.get("promotional"),
        neutral_path=output_config.get("neutral"),
        skipped_path=output_config.get("skipped"),
        num_pages=output_config.get("num_pages", -1),
    )


if __name__ == "__main__":
    main()
