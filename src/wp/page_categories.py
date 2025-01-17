import re
from dataclasses import dataclass


@dataclass
class PageCategories:
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
        max_len = max(len(prefix) for prefix in prefixes)
        string_lower = string[:max_len].lower()
        return any(string_lower.startswith(prefix.lower()) for prefix in prefixes)

    @classmethod
    def categorize(cls, id: int, title: str, text: str) -> "PageCategories":
        categories = cls()
        if id is None:
            categories.skip_missing_id = True
            return categories

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
            categories.skip_namespace = True
            return categories

        if cls._starts_with_any(text, {"#REDIRECT"}):
            categories.skip_redirect = True
            return categories

        if cls.disambiguation_re.search(text):
            categories.skip_disambiguation = True
            return categories

        good_replaced = cls.good_article_re.subn("", text)
        text = good_replaced[0]
        categories.good = good_replaced[1] > 0

        featured_replaced = cls.featured_article_re.subn("", text)
        text = featured_replaced[0]
        categories.featured = featured_replaced[1] > 0

        advert_replaced = cls.advert_re.subn("", text)
        text = advert_replaced[0]
        categories.advert = advert_replaced[1] > 0

        coi_replaced = cls.coi_re.subn("", text)
        text = coi_replaced[0]
        categories.coi = coi_replaced[1] > 0

        fanpov_replaced = cls.fanpov_re.subn("", text)
        text = fanpov_replaced[0]
        categories.fanpov = fanpov_replaced[1] > 0

        pr_replaced = cls.pr_re.subn("", text)
        text = pr_replaced[0]
        categories.pr = pr_replaced[1] > 0

        resume_replaced = cls.resume_re.subn("", text)
        text = resume_replaced[0]
        categories.resume = resume_replaced[1] > 0

        return categories
