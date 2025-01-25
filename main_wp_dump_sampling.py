import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


def write_sample(input_path: str, output_path: str, sample_size: int) -> None:
    """
    Writes a sample of lines from the input file to the output file using reservoir sampling.

    Args:
        input_path (str): The path to the input file.
        output_path (str): The path to the output file.
        sample_size (int): The number of lines to sample.
    """
    logging.info(
        f"Starting to write sample from {input_path} to {output_path} with sample size {sample_size}"
    )
    sampled_lines = reservoir_sampling(input_path, sample_size)
    with open(output_path, mode="w", newline="", encoding="utf-8") as output_file:
        output_file.writelines(sampled_lines)
    logging.info(f"Finished writing sample to {output_path}")


def reservoir_sampling(file_path: str, sample_size: int) -> list[str]:
    """
    Performs reservoir sampling on the input file and returns a sample of lines.

    Args:
        file_path (str): The path to the input file.
        sample_size (int): The number of lines to sample.

    Returns:
        list[str]: A list of sampled lines.
    """
    logging.info(
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
                logging.info(f"Processed {i} lines")
    logging.info(f"Finished reservoir sampling on {file_path}")
    return sample


def main() -> None:
    """
    Main function to write samples from multiple input files to output files.
    """
    write_sample("data/wp/good.csv", "data/wp/good_sample.csv", 30_000)
    write_sample("data/wp/promotional.csv", "data/wp/promotional_sample.csv", 30_000)
    write_sample("data/wp/neutral.csv", "data/wp/neutral_sample.csv", 30_000)


if __name__ == "__main__":
    main()
