# ------------------------------------------------------------------------------------------------------------------- #
# Utils file
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Union
import numpy as np
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


def generate_network_model(adj: np.array[:, :], edgelist: dict, blocks: dict) -> np.array[:, :]:
    """
    Generates a network with a given block structure
    """
    for (cl_i,cl_j) , nlinks in edgelist.items():
        for k in range(nlinks):
            i,j = create_link(cl_i, cl_j, adj, blocks)
            adj[i,j] = 1
    return adj


def create_link(cl_i: int, cl_j: int, adj: np.array[:, :], blocks: dict, recursive: bool = False) -> List[int, int]:
    """
    Returns the index of two nodes to be connected by a link
    :param cl_i: cluster of node i
    :param cl_j: cluster of node j
    :param adj: adjacency matrix of the in-the-making network
    :param blocks: block structure of the network
    :param recursive:
    :return:
    """
    i = np.random.choice(blocks[cl_i])
    j = np.random.choice(blocks[cl_j])
    if recursive:
        if adj[i,j] == 0:
            return i,j
        else:
            return create_link(cl_i,cl_j,adj,blocks)
    else:
        return i,j


def compute_growthrate(x,k=1):
    """
    Computes log-growthrates
    """
    return np.log(x).diff(k).replace([-np.inf,np.inf],np.nan)


def center_and_normalize(x):
    """
    Centers and normalizes the time-series
    """
    y = x-x.mean()
    N = y.count()
    var_proxy = (y.var()*(N-1)-y**2)/(N-2.)
    return y/np.sqrt(var_proxy)
