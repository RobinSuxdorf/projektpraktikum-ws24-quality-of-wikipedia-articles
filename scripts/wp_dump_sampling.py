"""Sample Wikipedia dump with reservoir sampling

Author: Johannes Krämer
"""

import logging
import sys
import random
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


def write_sample(input_path: str, output_path: str, sample_size: int) -> None:
    """
    Writes a sample of lines from the input file to the output file using reservoir sampling.

    Args:
        input_path (str): The path to the input file.
        output_path (str): The path to the output file.
        sample_size (int): The number of lines to sample.
    """
    logger.info(
        f"Starting to write sample from {input_path} to {output_path} with sample size {sample_size}"
    )
    sampled_lines = reservoir_sampling(input_path, sample_size)
    with open(output_path, mode="w", newline="", encoding="utf-8") as output_file:
        output_file.writelines(sampled_lines)
    logger.info(f"Finished writing sample to {output_path}")


def reservoir_sampling(file_path: str, sample_size: int) -> list[str]:
    """
    Performs reservoir sampling on the input file and returns a sample of lines.

    Args:
        file_path (str): The path to the input file.
        sample_size (int): The number of lines to sample.

    Returns:
        list[str]: A list of sampled lines.
    """
    logger.info(
        f"Starting reservoir sampling on {file_path} with sample size {sample_size}"
    )
    sample = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        header = file.readline()
        sample.append(header)
        for i, line in enumerate(
            file, start=1
        ):  # Start enumeration from 1 to account for the header
            if i <= sample_size:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sample[j + 1] = line  # +1 to account for the header
            if i % 10_000 == 0:
                logger.info(f"Processed {i} lines")
    logger.info(f"Finished reservoir sampling on {file_path}")
    return sample


def run_sampling(config: dict) -> None:
    """
    Run the sampling process for multiple input files.

    Args:
        config (dict): Configuration dictionary.
    """
    for sample_config in config.get("sample", []):
        write_sample(
            sample_config["input"],
            sample_config["output"],
            sample_config["sample_size"],
        )


def main() -> None:
    """
    Main function to write samples from multiple input files to output files.
    """
    parser = get_argument_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        run_sampling(config)
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
