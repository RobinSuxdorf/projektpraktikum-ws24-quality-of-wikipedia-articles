# main.py

import logging
import sys
from src import load_data
from src.utils import get_argument_parser, validate_file_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = get_argument_parser()
    args = parser.parse_args()

    try:
        validate_file_paths(args.good_file, args.promo_file)
        data = load_data(
            good_file_path=args.good_file,
            promo_file_path=args.promo_file,
            nrows=args.nrows,
            shuffle=args.shuffle,
        )
        logger.info("Data loaded successfully.")
        logger.info(f"First few rows of the loaded data:\n{data.head()}")
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
