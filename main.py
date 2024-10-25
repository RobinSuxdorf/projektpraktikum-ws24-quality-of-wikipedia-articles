import logging
from src import prepare_data

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    data = prepare_data("data/raw/promotional.csv", "data/raw/good.csv", 10)
    logging.info("Cleaned data:")
    logging.info(f"\n {data.head()}")
