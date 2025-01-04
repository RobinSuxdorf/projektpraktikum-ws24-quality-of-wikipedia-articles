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

        self.advert_re = re.compile(
            r"\{\{(Promotional|Ad|Advertising|Advertisement|Promotion|Promo|Promotional section|Advert section|promotion-inline)(\|[^\}]*)?\}\}"
        )
        self.coi_re = re.compile(r"\{\{COI(\|[^\}]*)?\}\}")
        self.fanpov_re = re.compile(r"\{\{Fan POV(\|[^\}]*)?\}\}")
        self.pr_re = re.compile(r"\{\{Cleanup press release(\|[^\}]*)?\}\}")
        self.resume_re = re.compile(
            r"\{\{(Resume-like|Like resume|Cleanup resume)(\|[^\}]*)?\}\}"
        )

    def write_to_csv(
        self,
        good_path: str,
        promotional_path: str,
        neutral_path: str,
        num_pages: int = -1,
    ):
        logging.info(
            "Input - WP dump: %s, index: %s",
            self.dump_path,
            self.index_path,
        )
        logging.info(
            "Output - good: %s, promotional: %s, neutral: %s (pages: %d)",
            good_path,
            promotional_path,
            neutral_path,
            num_pages,
        )
        with open(self.dump_path, "rb") as dump_file, open(
            good_path, "w", newline="", encoding="utf-8"
        ) as good_file, open(
            promotional_path, "w", newline="", encoding="utf-8"
        ) as promotional_file, open(
            neutral_path, "w", newline="", encoding="utf-8"
        ) as neutral_file:
            good_writer = csv.writer(good_file)
            promotional_writer = csv.writer(promotional_file)
            neutral_writer = csv.writer(neutral_file)
            self._write_to_csv(
                num_pages, dump_file, good_writer, promotional_writer, neutral_writer
            )

    def _write_to_csv(
        self, num_pages, dump_file, good_writer, promotional_writer, neutral_writer
    ):
        offsets = self._read_offsets()
        good_writer.writerow(["text"])
        promotional_writer.writerow(["text", "advert", "coi", "fanpov", "pr", "resume"])
        neutral_writer.writerow(["text"])
        total_offsets = len(offsets)
        pages_processed = 0
        for i, offset in enumerate(offsets):
            if num_pages > 0 and pages_processed >= num_pages:
                break
            dump_file.seek(offset)
            next_offset = offsets[i + 1] if i + 1 < len(offsets) else None
            chunk_size = next_offset - offset if next_offset else None
            logging.info(
                "Processing offset %d/%d (pages processed: %d)",
                i + 1,
                total_offsets,
                pages_processed,
            )
            compressed_data = dump_file.read(chunk_size)
            decompressed_data = bz2.decompress(compressed_data)
            root = self._parse_xml(decompressed_data)
            for page in root.findall(".//page"):
                pages_processed += 1
                if num_pages > 0 and pages_processed >= num_pages:
                    break
                self._process_page(
                    page, good_writer, promotional_writer, neutral_writer
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

    def _parse_xml(self, decompressed_data: bytes) -> ET.Element:
        wrapped_data = [
            "<root>",
            decompressed_data.decode("utf-8", errors="replace"),
            "</root>",
        ]
        return ET.fromstringlist(wrapped_data)

    def _process_page(
        self, page: ET.Element, good_writer, promotional_writer, neutral_writer
    ):
        text_elem = page.find(".//revision/text")
        title_elem = page.find(".//title")
        if (
            text_elem is not None
            and text_elem.text is not None
            and title_elem is not None
            and title_elem.text is not None
        ):
            self._process_text(
                title_elem.text,
                text_elem.text,
                good_writer,
                promotional_writer,
                neutral_writer,
            )

    def _process_text(
        self, title, text, good_writer, promotional_writer, neutral_writer
    ):
        if text.startswith("#REDIRECT [["):
            logging.info("skipping redirect: %s", title)
            return

        text = text.replace("\n", " ").replace("\r", " ")

        good_replaced = self.good_article_re.subn("", text)
        text = good_replaced[0]
        good_article = bool(good_replaced[1])

        featured_replaced = self.featured_article_re.subn("", text)
        text = featured_replaced[0]
        featured_article = bool(featured_replaced[1])

        advert_replaced = self.advert_re.subn("", text)
        text = advert_replaced[0]
        advert_article = bool(advert_replaced[1])

        coi_replaced = self.coi_re.subn("", text)
        text = coi_replaced[0]
        coi_article = bool(coi_replaced[1])

        fanpov_replaced = self.fanpov_re.subn("", text)
        text = fanpov_replaced[0]
        fanpov_article = bool(fanpov_replaced[1])

        pr_replaced = self.pr_re.subn("", text)
        text = pr_replaced[0]
        pr_article = bool(pr_replaced[1])

        resume_replaced = self.resume_re.subn("", text)
        text = resume_replaced[0]
        resume_article = bool(resume_replaced[1])

        promotional = bool(
            advert_article
            or coi_article
            or fanpov_article
            or pr_article
            or resume_article
        )
        if good_article or featured_article:
            logging.info("good: %s", title)
            good_writer.writerow([text])
        elif promotional:
            logging.info(
                "promotional (advert: %s, coi: %s, fanpov: %s, pr: %s, resume: %s): %s",
                advert_article,
                coi_article,
                fanpov_article,
                pr_article,
                resume_article,
                title,
            )
            promotional_writer.writerow(
                [
                    text,
                    advert_article,
                    coi_article,
                    fanpov_article,
                    pr_article,
                    resume_article,
                ]
            )
        else:
            logging.info("neutral: %s", title)
            neutral_writer.writerow([text])
