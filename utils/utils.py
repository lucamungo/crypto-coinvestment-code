# ------------------------------------------------------------------------------------------------------------------- #
# Utils file
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Union
import pandas as pd
import matplotlib as mpl


def setup_mpl() -> None:
    """
    :return:
    """
    mpl.rc("font", size=15)
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["axes.spines.top"] = False
    return None


def cleanUrls(url: str) -> str:
    """
    Cleans the URLs scraped from CoinMarketCap
    :param url: initial url
    :return: cleaned url
    """
    if len(url) == 0:
        return url
    else:
        clean_url = url.replace("[", "").replace("]", "").replace("'", "").split(",")
        return clean_url


def singleUrlToDomain(url: str) -> Union[str, None]:
    """
    Transforms a single Url address in a domain (example: https://www.crypto.com -> crypto.com)
    """
    if isinstance(url, str) is True:
        domain = url.replace("https", "http")
        domain = domain.replace("http://", "")
        domain = domain.replace("www.", "")
        domain = domain.replace(" ", "")
        domain = domain.split("/")[0]
        return domain
    else:
        return None


def urlsToDomain(urls: pd.DataFrame) -> List[str]:
    """
    Transforms urls to domains. Ignores linkedin, twitter, and github urls
    """
    all_domains = []
    for idx, row in urls.iterrows():

        for url in row["URL"]:
            if isinstance(url, str) is True:
                if ("linkedin" in url) or ("twitter" in url) or ("github" in url):
                    pass
                else:
                    domain = singleUrlToDomain(url)
                    all_domains.append(domain)

    return all_domains


def flagsafe(x, word):
    """
    Checks if a word is inside a string x
    :param x:
    :param word:
    :return:
    """
    if isinstance(x, str):
        if word in x:
            return True
        else:
            return False
    else:
        return False
