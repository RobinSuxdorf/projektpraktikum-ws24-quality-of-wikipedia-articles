import logging
import bz2
import csv
import re
import xml.etree.ElementTree as ET
from typing import List

def _read_offsets(index_path: str) -> List[int]:
    offsets = set()
    with bz2.open(index_path, 'rt', encoding='utf-8') as index_file:
        for line in index_file:
            parts = line.split(':')
            offset = int(parts[0])
            offsets.add(offset)
    return sorted(offsets)


def _decompress_data(compressed_data: bytes) -> bytes:
    return bz2.decompress(compressed_data)


def _parse_xml(decompressed_data: bytes, offset: int) -> ET.Element:
    wrapped_data = ["<root>", decompressed_data.decode('utf-8', errors='replace'), "</root>"]
    return ET.fromstringlist(wrapped_data)


def _process_page(page: ET.Element, good_article_re: re.Pattern, featured_article_re: re.Pattern, promotional_re: re.Pattern, curly_braces_re: re.Pattern) -> List:
    text_elem = page.find('.//revision/text')
    if text_elem is not None and text_elem.text is not None:
        text = text_elem.text.replace('\n', ' ').replace('\r', ' ')
        good_article = bool(good_article_re.search(text))
        featured_article = bool(featured_article_re.search(text))
        promotional = bool(promotional_re.search(text))
        text = curly_braces_re.sub('', text)
        if good_article or featured_article or promotional:
            logging.info("Article flags - good_article: %d, featured_article: %d, promotional: %d",
                         int(good_article), int(featured_article), int(promotional))
            return [int(good_article), int(featured_article), int(promotional), text]
    return []


def convert_wp_dump(dump_path: str, index_path: str, output_csv_path: str, num_pages: int = -1):
    """
    Extracts Wikipedia articles from a multistream dump, processes them, and writes the results to a CSV file.

    Parameters:
    dump_path (str): Path to the Wikipedia multistream dump file (bz2 format).
    index_path (str): Path to the corresponding index file (bz2 format).
    output_csv_path (str): Path to the output CSV file where the results will be written.
    num_pages (int): Number of pages to process. If set to a positive number, only the first num_pages pages will be processed.
                     If set to -1 (default), all pages will be processed.

    The CSV file will contain the article text and boolean fields indicating the presence of specific templates:
    - 'good_article': 1 if the article contains the {{good article}} template, 0 otherwise.
    - 'featured_article': 1 if the article contains the {{Featured article}} template, 0 otherwise.
    - 'promotional': 1 if the article contains the {{Promotional}} template, 0 otherwise.
    """
    logging.info("convert_wp_dump called with dump_path=%s, index_path=%s, output_csv_path=%s, num_pages=%d",
                 dump_path, index_path, output_csv_path, num_pages)

    # Regular expressions to match the templates and remove parts enclosed in double curly braces
    good_article_re = re.compile(r'\{\{good article.*?\}\}', re.IGNORECASE)
    featured_article_re = re.compile(r'\{\{Featured article.*?\}\}', re.IGNORECASE)
    promotional_re = re.compile(r'\{\{Promotional.*?\}\}', re.IGNORECASE)
    curly_braces_re = re.compile(r'\{\{.*?\}\}')

    # Read the index file and get the byte offsets
    offsets = _read_offsets(index_path)
    total_offsets = len(offsets)
    logging.info("Number of unique offsets read: %d", total_offsets)

    # Open the CSV file for writing
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['good_article', 'featured_article', 'promotional', 'text'])

        # Process each offset
        with open(dump_path, 'rb') as dump_file:
            pages_processed = 0
            for i, offset in enumerate(offsets):
                if num_pages > 0 and pages_processed >= num_pages:
                    break

                dump_file.seek(offset)
                next_offset = offsets[i + 1] if i + 1 < len(offsets) else None
                chunk_size = next_offset - offset if next_offset else None

                logging.info("Processing offset %d/%d: %d, chunk size: %s", i + 1, total_offsets, offset, chunk_size)
                compressed_data = dump_file.read(chunk_size)
                logging.info("Number of bytes read: %d", len(compressed_data))

                decompressed_data = _decompress_data(compressed_data)
                root = _parse_xml(decompressed_data, offset)

                for page in root.findall('.//page'):
                    if num_pages > 0 and pages_processed >= num_pages:
                        break

                    row = _process_page(page, good_article_re, featured_article_re, promotional_re, curly_braces_re)
                    if row:
                        writer.writerow(row)
                    pages_processed += 1

