# main.py

import logging
import sys
from src import load_data, get_argument_parser, validate_file_paths, load_config

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
        config = load_config(args.config)
        data_loader_config = config.get("data_loader")
        validate_file_paths(data_loader_config)
        data = load_data(data_loader_config)

        logger.info(f"First few rows of the loaded data:\n{data.head()}")
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
