# ------------------------------------------------------------------------------------------------------------------- #
# Python routine to generate (most of) the figures in the paper
# ------------------------------------------------------------------------------------------------------------------- #

import sys
import getopt
from environs import Env
import glob

import yaml
import pickle as pkl

import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering

from utils.utils import *

# Load environment variables
env = Env()
env.read_env("./.env", recurse=False)

PATH_CRUNCHBASE = env.str("PATH_CRUNCHBASE", None)
PATH_EXPORT = env.str("PATH_EXPORT", None)
FILE_EDGELIST = env.str("FILE_EDGELIST", None)

FILE_URLS = env.str("FILE_URLS", None)
FILE_TAGS = env.str("FILE_TAGS", None)
FILE_TICKS = env.str("FILE_TICKS", None)
CRYPTO_TO_REMOVE = env.str("CRYPTO_TO_REMOVE", None)
MKT_DATA = env.str("MKT_DATA", None)

TODAY = pd.to_datetime("today").date

# Define palette
blue_reds = ["#011f4b", "#03396c", "#005b96", "#6497b1", "#b3cde0", "#f15152", "#c3423f", "#95190c", "#702632"]
yellow = "#6C9A8B"


def setup_mpl() -> None:
    mpl.rc("font", size=16)
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False


def generate_figure_1() -> None:
    """
    Generates figure 1
    """
    # Load urls from CoinMarketCap. Urls are needed to merge Crunchbase and CoinMarketCap data
    urls = pd.read_csv(FILE_URLS)
    urls["URL"] = urls.URL.apply(lambda x: cleanUrls(x))

    # Split URLS in columns
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

    # Load Market data
    mktdata = pd.read_csv(MKT_DATA).drop(columns=["Unnamed: 0"])
    mktdata["date"] = pd.to_datetime(mktdata.date)
    mktdata["mcap"] = mktdata.mcap.apply(lambda x: x.replace(".", ""))
    mktdata["mcap"] = mktdata.mcap.apply(lambda x: x.replace(",", "."))
    mktdata["mcap"] = mktdata.mcap.astype(float) * 1e9
    mktdata = mktdata.loc[mktdata.date < "2021"]
    mktdata = mktdata.loc[mktdata.mcap != 0]

    funding_rounds["announced_on"] = pd.to_datetime(funding_rounds.announced_on)
    funding_rounds = funding_rounds.loc[funding_rounds.announced_on.dt.year > 2006]

    raised_amount = (
        funding_rounds.groupby(pd.to_datetime(funding_rounds.announced_on.dt.year.astype(str)))
        .raised_amount_usd.sum()
        .reset_index()
    )

    n_investments = (
        funding_rounds.groupby(pd.to_datetime(funding_rounds.announced_on.dt.year.astype(str)))
        .raised_amount_usd.nunique()
        .reset_index()
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    ax.plot(
        raised_amount.announced_on,
        raised_amount.raised_amount_usd,
        color="#005b96",
        marker="o",
        mfc="white",
        markersize=10,
        label="Raised amount",
    )

    ax.plot(
        mktdata.date,
        mktdata.mcap,
        color=yellow,
        # marker = "d",
        markersize=7,
        mfc="white",
        markevery=2,
        lw=1.5,
        label="Cryptocurrencies Market Cap",
    )

    ax1 = ax.twinx()

    ax1.plot(
        n_investments.announced_on,
        n_investments.raised_amount_usd,
        color="#f15152",
        marker="o",
        mfc="white",
        markersize=10,
        label="Number of investments",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Market Cap / Raised Amount ($)")
    ax1.set_ylabel("Number of investments")

    ax.set_yscale("log")
    ax1.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    labels = ["Raised amount", "Cryptocurrency Market Cap"]
    handles1, labels1 = ax1.get_legend_handles_labels()

    handles = [handles[0], handles1[0], handles[1]]
    labels = [labels[0], labels1[0], labels[1]]

    ax.legend(handles, labels, frameon=False, fontsize=20)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(PATH_EXPORT + "investments_vs_time.pdf", dpi=300)
    plt.show()
    return None


def generate_figure_2() -> None:
    """
    Generates the first panel
    """
    # Import edge list
    edge_list = pd.read_csv(FILE_EDGELIST, encoding="latin1")
    edge_list["announced_on"] = pd.to_datetime(edge_list.announced_on)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Panel 1
    # ---------------------------------------------------------------------------------------------------------------- #

    fig = plt.figure(figsize=(16, 9), dpi=200)
    gs = fig.add_gridspec(2, 2)

    ax = [None] * 3

    ax[0] = fig.add_subplot(gs[1, :])
    ax[1] = fig.add_subplot(gs[0, 0])
    ax[2] = fig.add_subplot(gs[0, 1])

    # Figure 1

    dates = ["2014", "2016", "2018", "2020", "2022"]

    xs = []
    ys = []
    degrees = []
    nedges = []
    nnodes = []

    max_x = 0
    max_y = 0

    for date in dates:
        dft = edge_list.loc[edge_list.announced_on < date].copy().drop(columns=["announced_on"])
        dft = dft.merge(dft, left_on="investor_name", right_on="investor_name").drop(columns="investor_name")
        nedges.append(dft.shape[0])
        nnodes.append(dft.ID_x.nunique())
        dft = dft.groupby("ID_x").ID_y.count().values
        degrees.append(dft)

        dft = Counter(dft)
        x = list(dft.keys())
        y = list(dft.values())

        idxs = np.argsort(x)
        x = np.array(x)[idxs]
        y = np.array(y)[idxs]

        max_x = max(max_x, max(x))
        max_y = max(max_y, max(y))

        xs.append(x)
        ys.append(y)

    bintype = "lin"

    if bintype == "lin":
        bins = np.linspace(0.1, max_x + 1, 30)
    else:
        bins = np.logspace(np.log(0.1), np.log(max_x + 1), 30)
    btp = (0.5 * (bins + np.roll(bins, -1)))[:-1]

    for i in range(len(degrees)):
        hist, _ = np.histogram(degrees[i], bins=bins)
        hist = hist / sum(hist)

        mask = hist != 0

        ax[0].plot(
            btp[mask],
            hist[mask],
            color=blue_reds[4 - i],
            lw=1 + 0.2 * i,
            marker="^",
            markersize=8,
            mfc="white",
            label=dates[i],
        )

    ax[0].set_yscale("log")
    ax[0].set_xscale("log")

    ax[0].set_xlabel("Degree")
    ax[0].set_ylabel("Frequency")

    ax[0].legend(ncol=2, edgecolor="none")

    # Figure 2 & 3

    dates = pd.date_range("2014", "2022", freq="3m")

    nedges = []
    nnodes = []

    for date in dates:
        dft = edge_list.loc[edge_list.announced_on < date].copy().drop(columns=["announced_on"])
        dft = dft.merge(dft, left_on="investor_name", right_on="investor_name").drop(columns="investor_name")

        nedges.append(dft.shape[0])
        nnodes.append(dft.ID_x.nunique())

    ax[1].plot(dates, nnodes, color=blue_reds[2], marker="o", markersize=8, mfc="white", label="Nodes")

    ax[2].plot(dates, nedges, color=blue_reds[5], marker="o", markersize=8, mfc="white", label="Edges")

    ax[1].set_yscale("log")
    ax[2].set_yscale("log")

    ax[1].set_ylabel("Nodes")
    ax[1].set_xlabel("Date")
    ax[2].set_ylabel("Edges")
    ax[2].set_xlabel("Date")

    ax[1].text(
        -0.1,
        2.3,
        "A",
        color="black",
        weight="bold",
        size=20,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
    )

    ax[2].text(
        0.5,
        2.3,
        "B",
        color="black",
        weight="bold",
        size=20,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
    )

    ax[0].text(
        -0.1,
        1.05,
        "C",
        color="black",
        weight="bold",
        size=20,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
    )

    plt.tight_layout()
    plt.savefig(PATH_EXPORT + "/nodes_edges_degree_distribution.pdf", dpi=300)
    plt.show()
    return None


def generate_figure_3():
    """
    In and out density plot figure
    :return:
    """
    # Import node list
    edge_list = pd.read_csv(FILE_EDGELIST, encoding="latin1")
    edge_list["announced_on"] = pd.to_datetime(edge_list.announced_on)

    # ---------------------------------------------------------------------------------------------------------------- #
    # In and out densities plot
    # ---------------------------------------------------------------------------------------------------------------- #

    dft = edge_list.loc[edge_list.announced_on < "2022"].copy().drop(columns=["announced_on"])
    dft = dft.merge(dft, left_on="investor_name", right_on="investor_name")
    dft = dft.groupby(["ID_x", "ID_y"], as_index=False).investor_name.count()
    dft = dft.rename(columns={"investor_name": "weight"})
    dft = dft.loc[dft.ID_x != dft.ID_y]

    g = nx.Graph()

    for idx, row in dft.iterrows():
        g.add_edge(row["ID_x"], row["ID_y"], weight=row["weight"])

    Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(Gcc[0])

    # Create tags dataset
    with open(FILE_TAGS, "rb") as infile:
        tags = pkl.load(infile)
    with open(FILE_TICKS, "rb") as infile:
        IDs = pkl.load(infile)

    allTags = set()
    for tag in tags:
        allTags = allTags.union(set(tag))
    allTags = list(allTags)

    data = []
    for t_of_coin in tags:
        data.append([t in t_of_coin for t in allTags])

    dftag = pd.DataFrame(data, columns=allTags)
    dftag["ID"] = IDs
    cols = [col for col in dftag.columns if ((col != "index") and (col != "ID"))]
    dftag = dftag.groupby("ID").agg(dict(zip(cols, [any] * len(cols))))

    # filter ownership-related tags
    cols_to_filter = [
        col for col in dftag.columns if ("portfolio" in col) or ("ventures" in col) or ("capital" in col)
    ]
    dftag = dftag.drop(columns=cols_to_filter)

    # only keep coins in the network
    addons = list(set(g.nodes()).difference(set(dftag.index)))
    data = np.zeros((len(addons), len(dftag.columns)))
    dfextra = pd.DataFrame(data, columns=dftag.columns)
    dfextra.index = addons
    dftag = pd.concat([dftag, dfextra])
    dftag = dftag.loc[list(g.nodes())]
    dftag = dftag.iloc[:, dftag.values.sum(axis=0) != 0]

    # Only keep coins with some tags
    coins_with_tags = dftag.values.sum(axis=1) != 0
    dftag = dftag.loc[coins_with_tags]

    # Get clusters
    n_clusters = 12
    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(dftag)

    # Get coinvestment adjacency matrix
    adj = nx.to_numpy_array(g)
    adj = (adj * (1 - np.eye(adj.shape[0])) > 0).astype(float)
    adj *= 1 - np.diag([np.nan] * adj.shape[0])
    adj = adj[:, coins_with_tags][coins_with_tags, :]

    # Random Benchmarks
    indensities_random = []
    outdensities_random = []

    for i in range(10_000):
        random_clusters = np.random.choice(clusters.max(), size=clusters.shape[0])
        for i in range(n_clusters):
            mask = random_clusters == i
            indensities_random.append(np.nanmean(adj[mask, :][:, mask]))
            outdensities_random.append(np.nanmean(adj[mask, :][:, ~mask]))

    # Empirical values
    indensities = []
    outdensities = []
    size = []

    for i in range(n_clusters):
        mask = clusters == i
        indensities.append(np.nanmean(adj[mask, :][:, mask]))
        outdensities.append(np.nanmean(adj[mask, :][:, ~mask]))
        size.append(2 * mask.sum())

    # Make figures
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Real values
    ax.scatter(indensities, outdensities, s=size, color=blue_reds[3], ec=blue_reds[0], zorder=3)
    # Diagonal ling
    ax.plot([0, 0.4], [0, 0.4], color=blue_reds[5], ls="--")

    # Random Benchmarks
    mask = np.ones(len(indensities_random)).astype(bool)

    sns.kdeplot(
        np.array(indensities_random)[mask],
        np.array(outdensities_random)[mask],
        color=blue_reds[6],
        fill=True,
        levels=20,
        alpha=0.4,
    )

    ax.set_xlabel("In-Cluster Density")
    ax.set_ylabel("Out-Cluster Density")

    # Annotations

    for i, (x, y) in enumerate(zip(indensities, outdensities)):

        mask = clusters == i
        label = (
            dftag.loc[mask].mean().sort_values(ascending=False).index.values[0]
            + "/"
            + dftag.loc[mask].mean().sort_values(ascending=False).index.values[1]
        )

        if "asset-management/" in label:
            dx, dy = -0.08, -0.017
        elif label in [
            "marketplace/payments",
            "payments/medium-of-exchange",
            "mineable/medium-of-exchange",
            "platform/smart-contracts",
        ]:
            dx, dy = 0.004, -0.008
        elif label == "defi/oracles":
            dx, dy = 0.003, 0.006
        elif label == "defi/governance":
            dx, dy = 0.004, 0.00
        elif "lending" in label:
            dx, dy = -0.065, 0.005
        elif "distributed" in label:
            dx, dy = 0.004, -0.005
        elif "collectibles" in label:
            dx, dy = 0.001, 0.006
        elif "media" in label:
            dx, dy = 0.005, -0.002
        else:
            dx, dy = 0.003, 0.001

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            size=6,
        )

    ax.set_ylim((None, 0.25))
    plt.savefig(PATH_EXPORT + "/in_and_out_densities.pdf", dpi=300)
    plt.show()
    return None


if __name__ == "__main__":

    arglist = sys.argv[1:]

    shortoptions = "a"
    longoptions = ["figure"]

    optlist, args = getopt.getopt(arglist, shortoptions, longoptions)

    figure_1 = False
    figure_2 = False
    figure_3 = False

    for o, a in optlist:
        if o == "-a":
            figure_1 = True
            figure_2 = True
            figure_3 = True

        if o == "--figure":
            if "1" in a:
                figure_1 = True
            if "2" in a:
                figure_2 = True
            if "3" in a:
                figure_3 = True

        # Change mpl params
        setup_mpl()

    if figure_1 is True:
        generate_figure_1()
    if figure_2 is True:
        generate_figure_2()
    if figure_3 is True:
        generate_figure_3()
