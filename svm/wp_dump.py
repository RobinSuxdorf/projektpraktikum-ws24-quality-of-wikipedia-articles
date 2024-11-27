import logging
import bz2
import csv
import re
import xml.etree.ElementTree as ET
from typing import List


class WikipediaDump:
    def __init__(self, dump_path: str, index_path: str):
        self.dump_path = dump_path
        self.index_path = index_path

        self.good_article_re = re.compile(r"\{\{Good article\}\}")
        self.featured_article_re = re.compile(r"\{\{Featured article\}\}")

        self.promotional_re = re.compile(
            r"\{\{(Promotional|Ad|Advertising|Advertisement|Promotion|Promo)(\|.*)?\}\}"
        )
        self.promotional_section_re = re.compile(r"\{\{Promotional section(\|.*)?\}\}")
        self.press_release_re = re.compile(r"\{\{Cleanup press release(\|.*)?\}\}")
        self.promotion_inline_re = re.compile(r"\{\{promotion-inline(\|.*)?\}\}")
        self.fan_pov_re = re.compile(r"\{\{Fan POV(\|.*)?\}\}")
        self.resume_re = re.compile(
            r"\{\{(Resume-like|Like resume|Cleanup resume)(\|.*)?\}\}"
        )

        self.curly_braces_re = re.compile(r"\{\{.*?\}\}")

    def write_to_csv(self, output_csv_path: str, num_pages: int = -1):
        logging.info(
            "Writing WP dump with dump_path=%s, index_path=%s to output_csv_path=%s (num_pages=%d)",
            self.dump_path,
            self.index_path,
            output_csv_path,
            num_pages,
        )
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            with open(self.dump_path, "rb") as dump_file:
                self._write_to_csv(num_pages, dump_file, writer)

    def _write_to_csv(self, num_pages, dump_file, writer):
        offsets = self._read_offsets()
        writer.writerow(["good_article", "featured_article", "promotional", "text"])
        total_offsets = len(offsets)
        pages_processed = 0
        for i, offset in enumerate(offsets):
            if num_pages > 0 and pages_processed >= num_pages:
                break
            dump_file.seek(offset)
            next_offset = offsets[i + 1] if i + 1 < len(offsets) else None
            chunk_size = next_offset - offset if next_offset else None
            logging.info("Processing offset %d/%d (pages processed: %d)", i + 1, total_offsets, pages_processed)
            compressed_data = dump_file.read(chunk_size)
            decompressed_data = bz2.decompress(compressed_data)
            root = self._parse_xml(decompressed_data)
            for page in root.findall(".//page"):
                pages_processed += 1
                if num_pages > 0 and pages_processed >= num_pages:
                    break
                row = self._process_page(page)
                if row:
                    writer.writerow(row)

    def _read_offsets(self) -> List[int]:
        offsets = set()
        with bz2.open(self.index_path, "rt", encoding="utf-8") as index_file:
            for line in index_file:
                parts = line.split(":")
                offset = int(parts[0])
                offsets.add(offset)
        logging.info("Number of unique offsets read: %d", len(offsets))
        return sorted(offsets)

    def _parse_xml(self, decompressed_data: bytes) -> ET.Element:
        wrapped_data = [
            "<root>",
            decompressed_data.decode("utf-8", errors="replace"),
            "</root>",
        ]
        return ET.fromstringlist(wrapped_data)

    def _process_page(self, page: ET.Element) -> List:
        text_elem = page.find(".//revision/text")
        if text_elem is not None and text_elem.text is not None:
            text = text_elem.text.replace("\n", " ").replace("\r", " ")
            good_article = bool(self.good_article_re.search(text))
            featured_article = bool(self.featured_article_re.search(text))
            promotional = bool(
                self.promotional_re.search(text)
                or self.promotional_section_re.search(text)
                or self.press_release_re.search(text)
                or self.promotion_inline_re.search(text)
                or self.fan_pov_re.search(text)
                or self.resume_re.search(text)
            )
            text = self.curly_braces_re.sub("", text)
            if good_article or featured_article or promotional:
                logging.info(
                    "Article flags - good_article: %d, featured_article: %d, promotional: %d",
                    int(good_article),
                    int(featured_article),
                    int(promotional),
                )
                return [
                    int(good_article),
                    int(featured_article),
                    int(promotional),
                    text,
                ]
        return []
