"""Module for categorizing Wikipedia pages with quality and type classification functionalities.

Author: Johannes Krämer
"""

import re
from dataclasses import dataclass


@dataclass
class CategorizedPage:
    """
    Represents a categorized Wikipedia page with various attributes indicating
    the type and quality of the page content.
    """

    id: int = None
    title: str = ""
    text: str = ""
    good: bool = False
    featured: bool = False
    advert: bool = False
    coi: bool = False
    fanpov: bool = False
    pr: bool = False
    resume: bool = False
    skip_missing_id: bool = False
    skip_namespace: bool = False
    skip_redirect: bool = False
    skip_disambiguation: bool = False

    good_article_re = re.compile(r"\{\{(g|G)ood article\}\}")
    featured_article_re = re.compile(r"\{\{(f|F)eatured article\}\}")
    advert_re = re.compile(
        r"\{\{((p|P)romotional|(p|P)romotional tone|(A|a)d|(A|a)dvert|(A|a)dvertising|(A|a)dvertisement|(p|P)romotion|(p|P)romo|(p|P)romotional section|(A|a)dvert section|(p|P)romotion-inline)(\|[^\}]*)?\}\}"
    )
    coi_re = re.compile(r"\{\{(c|C)OI(\|[^\}]*)?\}\}")
    fanpov_re = re.compile(r"\{\{(f|F)an POV(\|[^\}]*)?\}\}")
    pr_re = re.compile(r"\{\{(c|C)leanup press release(\|[^\}]*)?\}\}")
    resume_re = re.compile(
        r"\{\{((r|R)esume-like|(r|R)esume like|(l|L)ike resume|(c|C)leanup resume)(\|[^\}]*)?\}\}"
    )
    disambiguation_re = re.compile(
        r"\{\{((d|D)isambiguation|(d|D)isambig|(d|D)ab)(\|[^\}]*)?\}\}"
    )

    @staticmethod
    def _starts_with_any(string: str, prefixes: set) -> bool:
        """
        Checks if the given string starts with any of the specified prefixes ignoring case.

        Args:
            string (str): The string to check.
            prefixes (set): A set of prefixes to check against.

        Returns:
            bool: True if the string starts with any of the prefixes, False otherwise.
        """
        max_len = max(len(prefix) for prefix in prefixes)
        string_lower = string[:max_len].lower()
        return any(string_lower.startswith(prefix.lower()) for prefix in prefixes)

    @classmethod
    def categorize(cls, id: int, title: str, text: str) -> "CategorizedPage":
        """
        Categorizes a Wikipedia page based on its content.

        Args:
            id (int): The ID of the page.
            title (str): The title of the page.
            text (str): The text content of the page.

        Returns:
            CategorizedPage: An instance of CategorizedPage with categorized attributes.
        """
        result = cls()
        result.id = id
        result.title = title
        result.text = text.replace("\n", " ").replace("\r", " ")

        if id is None:
            result.skip_missing_id = True
            return result

        if cls._starts_with_any(
            title,
            {
                "User:",
                "Wikipedia:",
                "File:",
                "MediaWiki:",
                "Template:",
                "Help:",
                "Category:",
                "Portal:",
                "Draft:",
                "MOS:",
                "TimedText:",
                "Module:",
            },
        ):
            result.skip_namespace = True
            return result

        if cls._starts_with_any(result.text, {"#REDIRECT"}):
            result.skip_redirect = True
            return result

        if cls.disambiguation_re.search(result.text):
            result.skip_disambiguation = True
            return result

        good_replaced = cls.good_article_re.subn("", result.text)
        result.text = good_replaced[0]
        result.good = good_replaced[1] > 0

        featured_replaced = cls.featured_article_re.subn("", result.text)
        result.text = featured_replaced[0]
        result.featured = featured_replaced[1] > 0

        advert_replaced = cls.advert_re.subn("", result.text)
        result.text = advert_replaced[0]
        result.advert = advert_replaced[1] > 0

        coi_replaced = cls.coi_re.subn("", result.text)
        result.text = coi_replaced[0]
        result.coi = coi_replaced[1] > 0

        fanpov_replaced = cls.fanpov_re.subn("", result.text)
        result.text = fanpov_replaced[0]
        result.fanpov = fanpov_replaced[1] > 0

        pr_replaced = cls.pr_re.subn("", result.text)
        result.text = pr_replaced[0]
        result.pr = pr_replaced[1] > 0

        resume_replaced = cls.resume_re.subn("", result.text)
        result.text = resume_replaced[0]
        result.resume = resume_replaced[1] > 0

        return result
