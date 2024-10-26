import logging
from src import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

if __name__ == "__main__":
    promotional_file = "data/raw/promotional.csv"
    good_file = "data/raw/good.csv"
    nrows = 10

    data = prepare_data(promotional_file, good_file, nrows)
    logging.info("Cleaned data:\n%s", data.head())
