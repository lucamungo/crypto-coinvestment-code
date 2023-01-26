# ------------------------------------------------------------------------------------------------------------------- #
# Python routine to create the cryptocurrency co-investment network
# ------------------------------------------------------------------------------------------------------------------- #
import sys
import getopt
from environs import Env
import glob

import yaml
from utils.utils import *

# Load environment variables
env = Env()
env.read_env("./.env", recurse=False)

PATH_CRUNCHBASE = env.str("PATH_CRUNCHBASE", "./")
PATH_EXPORTS = env.str("PATH_EXPORTS", "./")
FILE_URLS = env.str("FILE_URLS", "./data/")
INVESTORS_LIST = env.str("INVESTORS_LIST", None)
CRYPTO_TO_REMOVE = env.str("CRYPTO_TO_REMOVE", "./data/crypto_to_remove.yaml")

TODAY = pd.to_datetime("today").date

if __name__ == "__main__":

    arglist = sys.argv[1:]

    shortoptions = "v"
    longoptions = ["verbose"]

    optlist, args = getopt.getopt(arglist, shortoptions, longoptions)

    verbose = False

    for o, a in optlist:
        if (o == "-v") or (o == "--verbose"):
            verbose = True

    verboseprint = print if verbose else lambda *a, **k: None

    # Change matplotlib settings
    setup_mpl()
    allfiles = sorted(glob.glob(PATH_CRUNCHBASE + "*"))

    # Load urls from CoinMarketCap. Urls are needed to merge Crunchbase and CoinMarketCap data
    verboseprint("\t- Loading CoinMarketCap Urls")
    urls = pd.read_csv(FILE_URLS)
    urls["URL"] = urls.URL.apply(lambda x: cleanUrls(x))

    # Split URLS in columns
    verboseprint("\t- Processing Urls")
    urls_list = urls.URL.apply(lambda x: x if isinstance(x, list) else [x]).apply(lambda x: x + [None] * (4 - len(x)))
    urls_table = pd.DataFrame.from_records(urls_list)
    urls_table.columns = [f"URL{i}" for i in range(1, 5)]
    urls = pd.concat([urls, urls_table], axis=1)

    # Remove all the cryptocurrencies that are not relevant for the analysis: stablecoins, synthetic tokens, derivatives
    #  wrapped coins, etc.

    # Getting out all the mirrored stocks
    urls["flag"] = urls.NAME.apply(lambda x: flagsafe(x, "Mirrored"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    urls["flag"] = urls.NAME.apply(lambda x: flagsafe(x, "tokenized"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    urls["flag"] = urls.NAME.apply(lambda x: flagsafe(x, "Amun"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    # Remove derivatives
    urls["flag"] = urls.ID.apply(lambda x: flagsafe(x, "BEAR"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    urls["flag"] = urls.ID.apply(lambda x: flagsafe(x, "BULL"))
    urls.loc[urls.NAME == "BULL FINANCE", "flag"] = False
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    urls["flag"] = urls.URL1.apply(lambda x: flagsafe(x, "leveraged-tokens"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    urls["flag"] = urls.URL1.apply(lambda x: flagsafe(x, "aavegotchi"))
    urls = urls.loc[(~urls.flag) & (urls.ID != "GHST")].drop(columns=["flag"])

    # Remove stablecoins
    urls["flag"] = urls.ID.apply(lambda x: flagsafe(x, "USD"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    urls["flag"] = urls.ID.apply(lambda x: flagsafe(x, "EUR"))
    urls = urls.loc[~urls.flag].drop(columns=["flag"])

    # Other crypto to remove: stablecoins, wrapped coins, etc.
    with open(CRYPTO_TO_REMOVE, "r") as infile:
        crypto_to_remove = yaml.safe_load(infile)

    urls = urls.loc[~urls.ID.isin(crypto_to_remove["stablecoins"])]
    urls = urls.loc[~urls.ID.isin(crypto_to_remove["wrapped_coins"])]
    urls = urls.loc[~urls.ID.isin(crypto_to_remove["others"])]

    # Transform Urls in domains
    domains = urlsToDomain(urls)

    # Filter urls to remove wikipedia, linkedin, twitter, etc.
    domains = [url for url in domains if "wikipedia" not in url]
    domains = [url for url in domains if "linkedin" not in url]
    domains = [url for url in domains if "bit.ly" not in url]
    domains = [url for url in domains if "medium.com" not in url]
    domains = [url for url in domains if "t.me" not in url]
    domains = [url for url in domains if "play.google.com" not in url]

    # Manual fix for some urls
    domains += [
        "balancer.fi",
        "bitcoin.org",
        "bitcoincash.org",
        "cardanofoundation.org",
        "celsius.network",
        "chiliz.io",
        "cosmos.network",
        "crypto.com",
        "curve.fi",
        "digibyte.co",
        "etclabs.org",
        "eos.global",
        "funfair.io",
        "huobi.com",
        "icon.foundation",
        "on.wax.io",
        "ripple.com",
        "rsk.co",
        "trusttoken.com",
        "uquid.com",
    ]
    domains = [url for url in domains if url != "etherscan.io"]

    # Filter organizations with matching Urls
    verboseprint("\t- Loading Crunchbase Organizations")
    organizations_files = sorted(glob.glob(PATH_CRUNCHBASE + "/organizations*"))
    organizations = []
    for file in organizations_files:
        dfTemp = pd.read_csv(file)
        # Filter organizations file to keep only firms associated with CoinMarketCap cryptocurrencies
        dfTemp = dfTemp.loc[(dfTemp.domain.isin(domains))]
        organizations.append(dfTemp)

    organizations = pd.concat(organizations)

    # Match organizations with their Coinmarketcap tick
    urls_melted = (
        urls.drop(columns="URL").melt(id_vars=["NAME", "ID", "SLUG"], var_name="0", value_name="URL").drop(columns="0")
    )
    urls_melted.URL = urls_melted.URL.apply(singleUrlToDomain)
    urls_melted.drop_duplicates(inplace=True)
    organizations_with_ticks = pd.merge(
        organizations, urls_melted[["URL", "ID"]], how="left", left_on="domain", right_on="URL"
    )

    # Load investments and funding rounds data
    verboseprint("\t- Loading Investment data")
    investments = pd.read_csv(PATH_CRUNCHBASE + "investments.csv")
    funding_rounds = pd.read_csv(PATH_CRUNCHBASE + "funding_rounds.csv")

    investments = investments.merge(
        funding_rounds[["uuid", "org_name"]], how="left", left_on="funding_round_uuid", right_on="uuid"
    )
    investments = investments.loc[investments["org_name"].isin(organizations.name.values)]
    investments.loc[investments.investor_name.isin(organizations.name.unique()), "investor_name"] += "_i"

    investments = investments.merge(
        organizations_with_ticks[["name", "ID"]], how="left", left_on="org_name", right_on="name"
    )

    funding_rounds = funding_rounds.loc[funding_rounds["uuid"].isin(investments.funding_round_uuid.unique())]

    # Create investment network edgelist
    verboseprint("\t- Creating edgelist")
    edgelist = investments[["investor_name", "ID", "funding_round_uuid"]].merge(
        funding_rounds[["uuid", "announced_on"]], left_on="funding_round_uuid", right_on="uuid", how="left"
    )
    edgelist = (
        edgelist.sort_values(by="announced_on")
        .groupby(["investor_name", "ID"], as_index=False)
        .agg({"announced_on": "first"})
    )
    edgelist.to_csv(PATH_EXPORTS + "/network_time_edgelist.csv", index=False)
    verboseprint("\t- Done")
