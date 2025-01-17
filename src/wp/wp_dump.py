from collections import Counter
import logging
import bz2
import csv
import time
import xml.etree.ElementTree as ET
from typing import List

from src.wp.page_categories import PageCategories


class WikipediaDump:
    def __init__(self, dump_path: str, index_path: str):
        self.dump_path = dump_path
        self.index_path = index_path

    def write_to_csv(
        self,
        good_path: str,
        promotional_path: str,
        neutral_path: str,
        skipped_path: str,
        num_pages: int = -1,
    ):
        logging.info(
            "Input - WP dump: %s, index: %s",
            self.dump_path,
            self.index_path,
        )
        logging.info(
            "Output - good: %s, promotional: %s, neutral: %s, skipped: %s (pages: %d)",
            good_path,
            promotional_path,
            neutral_path,
            skipped_path,
            num_pages,
        )
        with open(self.dump_path, "rb") as dump_file, open(
            good_path, "w", newline="", encoding="utf-8"
        ) as good_file, open(
            promotional_path, "w", newline="", encoding="utf-8"
        ) as promotional_file, open(
            neutral_path, "w", newline="", encoding="utf-8"
        ) as neutral_file, open(
            skipped_path, "w", newline="", encoding="utf-8"
        ) as skipped_file:
            good_writer = csv.writer(good_file)
            promotional_writer = csv.writer(promotional_file)
            neutral_writer = csv.writer(neutral_file)
            skipped_writer = csv.writer(skipped_file)
            self._write_to_csv(
                num_pages,
                dump_file,
                good_writer,
                promotional_writer,
                neutral_writer,
                skipped_writer,
            )

    def _write_to_csv(
        self,
        num_pages,
        dump_file,
        good_writer,
        promotional_writer,
        neutral_writer,
        skipped_writer,
    ):
        offsets = self._read_offsets()
        good_writer.writerow(["id", "title", "text"])
        promotional_writer.writerow(
            ["id", "title", "text", "advert", "coi", "fanpov", "pr", "resume"]
        )
        neutral_writer.writerow(["id", "title", "text"])
        skipped_writer.writerow(["id", "title", "reason"])
        total_offsets = len(offsets)
        pages_processed = 0
        cnt = Counter()
        start = time.monotonic_ns()
        for i, offset in enumerate(offsets):
            start_offset = time.monotonic_ns()
            if num_pages > 0 and pages_processed >= num_pages:
                break
            dump_file.seek(offset)
            next_offset = offsets[i + 1] if i + 1 < len(offsets) else None
            chunk_size = next_offset - offset if next_offset else None
            compressed_data = dump_file.read(chunk_size)
            decompressed_data = bz2.decompress(compressed_data)
            root = self._parse_xml(decompressed_data)
            for page in root.findall(".//page"):
                type = self._process_page(
                    page,
                    good_writer,
                    promotional_writer,
                    neutral_writer,
                    skipped_writer,
                )
                cnt[type] += 1
                pages_processed += 1
                if num_pages > 0 and pages_processed >= num_pages:
                    break
            end_offset = time.monotonic_ns()
            logging.info(
                "Offset %d/%d finished in %d ns, avg %d ns (pages: %d, %s)",
                i + 1,
                total_offsets,
                end_offset - start_offset,
                (end_offset - start) / (i + 1),
                pages_processed,
                cnt,
            )

    def _read_offsets(self) -> List[int]:
        offsets = set()
        with bz2.open(self.index_path, "rt", encoding="utf-8") as index_file:
            for line in index_file:
                parts = line.split(":")
                offset = int(parts[0])
                offsets.add(offset)
        logging.info("Number of unique offsets read: %d", len(offsets))
        return sorted(offsets)
        # return sorted(offsets)[-10:]  # Return only the last 10 offsets for testing

    def _parse_xml(self, decompressed_data: bytes) -> ET.Element:
        decoded = decompressed_data.decode("utf-8", errors="replace")
        if decoded.startswith("<mediawiki"):
            wrapped_data = [
                decoded,
                "</mediawiki>",
            ]
        elif decoded.endswith("</mediawiki>"):
            wrapped_data = [
                "<mediawiki>",
                decoded,
            ]
        else:
            wrapped_data = [
                "<mediawiki>",
                decoded,
                "</mediawiki>",
            ]
        try:
            return ET.fromstringlist(wrapped_data)
        except ET.ParseError as e:
            logging.error(f"Error parsing XML: {e}")
            print(wrapped_data[1])
            raise e

    def _process_page(
        self,
        page: ET.Element,
        good_writer,
        promotional_writer,
        neutral_writer,
        skipped_writer,
    ) -> str:
        id_elem = page.find(".//id")
        id = id_elem.text if id_elem is not None else None

        title_elem = page.find(".//title")
        if title_elem is not None and title_elem.text is not None:
            title = title_elem.text
        else:
            title = "missing_title"

        text_elem = page.find(".//revision/text")
        if text_elem is not None and text_elem.text is not None:
            text = text_elem.text
        else:
            text = "missing_text"

        return self._process_text(
            id,
            title,
            text,
            good_writer,
            promotional_writer,
            neutral_writer,
            skipped_writer,
        )

    def _process_text(
        self,
        id,
        title,
        text,
        good_writer,
        promotional_writer,
        neutral_writer,
        skipped_writer,
    ) -> str:
        categories = PageCategories.categorize(id, title, text)
        if categories.skip_missing_id:
            logging.debug("%d - skipping missing ID: %s", 0, title)
            skipped_writer.writerow([0, title, "missing_id"])
            return "skipped"
        if categories.skip_namespace:
            logging.debug("%d - skipping special namespace: %s", id, title)
            skipped_writer.writerow([id, title, "namespace"])
            return "skipped"
        if categories.skip_redirect:
            logging.debug("%d - skipping redirect: %s", id, title)
            skipped_writer.writerow([id, title, "redirect"])
            return "skipped"
        if categories.skip_disambiguation:
            logging.debug("%d - skipping disambiguation: %s", id, title)
            skipped_writer.writerow([id, title, "disambiguation"])
            return "skipped"

        text = text.replace("\n", " ").replace("\r", " ")

        promotional = bool(
            categories.advert
            or categories.coi
            or categories.fanpov
            or categories.pr
            or categories.resume
        )
        if categories.good or categories.featured:
            logging.debug("%d - good: %s", id, title)
            good_writer.writerow([id, title, text])
            return "good"
        elif promotional:
            logging.debug(
                "%d - promotional (advert: %s, coi: %s, fanpov: %s, pr: %s, resume: %s): %s",
                id,
                categories.advert,
                categories.coi,
                categories.fanpov,
                categories.pr,
                categories.resume,
                title,
            )
            promotional_writer.writerow(
                [
                    id,
                    title,
                    text,
                    int(categories.advert),
                    int(categories.coi),
                    int(categories.fanpov),
                    int(categories.pr),
                    int(categories.resume),
                ]
            )
            return "promotional"
        else:
            logging.debug("%d - neutral: %s", id, title)
            neutral_writer.writerow([id, title, text])
            return "neutral"
