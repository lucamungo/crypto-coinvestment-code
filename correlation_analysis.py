# ------------------------------------------------------------------------------------------------------------------- #
# Analysis of correlations over the co-inv network + benchmarks
# ------------------------------------------------------------------------------------------------------------------- #
from environs import Env
import pickle as pkl

import numpy as np
import pandas as pd
import random

import networkx as nx
from collections import Counter

from utils.utils import *

# Load environment variables
env = Env()
env.read_env("./.env", recurse=False)

PATH_EXPORT = env.str("PATH_EXPORT", None)
FILE_NETWORK = env.str("NETWORK_FILE", None)
FILE_TIME_SERIES = env.str("TIME_SERIES_FILE", None)
FILE_CLUSTERS = env.str("CLUSTERS_FILE", None)


if __name__ == "__main__":

    # Load network
    with open(FILE_NETWORK, "rb") as inputfile:
        g = pkl.load(inputfile)
    adj_pd = nx.to_pandas_adjacency(g)

    # Load Crypto timeseries
    df = pd.read_csv(FILE_TIME_SERIES).rename(columns={"Symbol": "ID", "Price": "Close", "Date": "DATE"})
    df = df.loc[df.ID.isin(list(adj_pd.columns))]

    df["close_g"] = df.groupby("ID").Close.apply(compute_growthrate)
    df["close_g_r"] = df.groupby("ID").close_g.transform(center_and_normalize)

    df.drop_duplicates(subset=["DATE", "ID"], keep=False, inplace=True)

    stacked = df[["ID", "DATE", "close_g_r"]].sort_values(by=["ID", "DATE"])
    stacked.set_index(["DATE", "ID"], inplace=True)

    # Compute log-returns
    returns = stacked.unstack()
    returns.columns = returns.columns.get_level_values(1)
    for col in returns.columns:
        returns[col] = returns[col].apply(lambda x: np.nan if x > 2 else x)

    # Compute correlation matrix
    c = returns.corr(method="spearman", min_periods=60)
    adjClipped = adj_pd.loc[adj_pd.index.isin(c.columns)]
    adjClipped = adjClipped[[col for col in adjClipped.columns if col in c.columns]]
    adjClipped = adjClipped[c.columns].loc[c.columns]

    # Clean from marketmode
    w, v = np.linalg.eigh(c.fillna(0).values)  # TODO: Luca: fix here
    returnsclean = returns.subtract(np.nanmean(returns.values, axis=1), axis=0)
    cclean = returnsclean.corr()

    # Load blocks for blockmodel
    with open(FILE_CLUSTERS, "r") as infile:
        clusters = pkl.load(infile)
    blocks = pd.DataFrame.from_dict(dict(zip(["node", "cluster"], [np.arange(c.shape[0]), clusters])))
    blocks = blocks.groupby("cluster").node.apply(list).to_dict()
    blockmap = dict(zip(np.arange(c.shape[0]), clusters))

    links = np.where(adjClipped == 1)
    links = [(blockmap[x], blockmap[y]) for x, y in zip(links[0], links[1])]
    links = dict(Counter(links))

    # Create edgelist and links for "block" model
    blocks = pd.DataFrame.from_dict(dict(zip(["node", "cluster"], [np.arange(c.shape[0]), clusters])))
    blocks = blocks.groupby("cluster").node.apply(list).to_dict()
    blockmap = dict(zip(np.arange(c.shape[0]), clusters))

    links = np.where(adjClipped == 1)
    links = [(blockmap[x], blockmap[y]) for x, y in zip(links[0], links[1])]
    links = dict(Counter(links))

    # ---------------------------------------------------------------------------------------------------------------- #
    # Benchmarks
    # ---------------------------------------------------------------------------------------------------------------- #

    # Start here
    remove_seen_connections = True

    corrConnected = np.zeros((2, 5))

    corrCMMean = np.zeros((2, 5))
    corrCMStd = np.zeros((2, 5))

    corrBMMean = np.zeros((2, 5))
    corrBMStd = np.zeros((2, 5))

    corrERMean = np.zeros((2, 5))
    corrERStd = np.zeros((2, 5))

    densitiesCM = np.zeros((2, 5))
    distancesCM = np.zeros((2, 5))

    densitiesBM = np.zeros((2, 5))
    distancesBM = np.zeros((2, 5))

    densitiesER = np.zeros((2, 5))
    distancesER = np.zeros((2, 5))

    allcorrTr = []
    allcorrCM = []
    allcorrBM = []
    allcorrER = []

    allcorrTr_c = []
    allcorrCM_c = []
    allcorrBM_c = []
    allcorrER_c = []

    adjs = []

    adjnan = np.where(adjClipped.values == 0, np.nan, adjClipped.values)
    for i in range(adjnan.shape[0]):
        adjnan[i, i] = np.nan

    for k in range(1, 6):
        adj = (np.linalg.matrix_power(adjClipped.values, k) > 0).astype(float)

        if remove_seen_connections:
            for step in range(k - 1, 0, -1):
                adj -= (np.linalg.matrix_power(adjClipped.values, step) > 0).astype(float)
                adj = (adj > 0).astype(float)

        adj_store = adj.copy()
        adj = np.where(adj == 0, np.nan, adj)
        for i in range(adj.shape[0]):
            adj[i, i] = np.nan
        adjs.append(adj)

        corrConnected[0, k - 1] = np.nanmean(adj * c.values)
        corrConnected[1, k - 1] = np.nanmean(adj * cclean.values)

        if k == 1:
            allcorrTr.append(cclean.values[~np.isnan(adj * c.values)])
            allcorrTr_c.append(cclean.values[~np.isnan(adj * cclean.values)])

        p = np.nansum(adj) / (adj.shape[0] ** 2)

        corrCM = []
        distCM = []
        densityCM = []

        corrBM = []
        distBM = []
        densityBM = []

        corrER = []
        distER = []
        densityER = []

        corrCM_c = []
        distCM_c = []
        densityCM_c = []

        corrBM_c = []
        distBM_c = []
        densityBM_c = []

        corrER_c = []
        distER_c = []
        densityER_c = []

        counter = 0

        for n in range(1_000):
            # Configuration model
            degree_dist = (adj_store * (1 - np.eye(adj_store.shape[0]))).sum(axis=1).astype(int)
            adjrand = nx.to_numpy_array(nx.generators.degree_seq.configuration_model(degree_dist))
            adjrand = (adjrand > 0).astype(int)

            for i in range(adjrand.shape[0]):
                adjrand[i, i] = 0
            densityCM.append(np.nanmean(adjrand))
            adjrand = np.where(adjrand == 0, np.nan, adjrand)

            distCM.append(np.nansum(adj * adjrand) / np.nansum(adj))
            corrCM.append(np.nanmean(adjrand * c.values))

            distCM_c.append(np.nansum(adj * adjrand) / np.nansum(adj))
            corrCM_c.append(np.nanmean(adjrand * cclean.values))

            if k == 1:
                allcorrCM.append(cclean.values[~np.isnan(adjrand * c.values)])
                allcorrCM_c.append(cclean.values[~np.isnan(adjrand * cclean.values)])

            # "Block" model
            try:
                adjrand = np.zeros(adjClipped.shape)
                adjrand = generate_network_model(adj_store - np.eye(adj_store.shape[0]) * adj_store, links, blocks)
                densityBM.append(np.nanmean(adjrand))
                densityBM_c.append(np.nanmean(adjrand))

                distBM.append(np.nansum(adj_store * adjrand) / np.nansum(adj_store))

                adjrand = np.where(adjrand == 0, np.nan, adjrand)
                corrBM.append(np.nanmean(adjrand * c.values))

                distBM_c.append(np.nansum(adj * adjrand) / np.nansum(adj))
                corrBM_c.append(np.nanmean(adjrand * cclean.values))

                if k == 1:
                    allcorrBM.append(cclean.values[~np.isnan(adjrand * c.values)])
                    allcorrBM_c.append(cclean.values[~np.isnan(adjrand * cclean.values)])

                counter += 1
            except:
                pass

            # Erdos-Renyi
            adjrand = (np.random.random(size=adjnan.shape) < p).astype(int)
            for i in range(adjrand.shape[0]):
                adjrand[i, i] = 0
            densityER.append(np.mean(adjrand))
            densityER_c.append(np.mean(adjrand))
            adjrand = np.where(adjrand == 0, np.nan, adjrand)

            corrER.append(np.nanmean(adjrand * c.values))
            distER.append(np.nansum(adj * adjrand) / np.nansum(adj))

            corrER_c.append(np.nanmean(adjrand * cclean.values))
            distER_c.append(np.nansum(adj * adjrand) / np.nansum(adj))

            if k == 1:
                allcorrER.append(cclean.values[~np.isnan(adjrand * c.values)])
                allcorrER_c.append(cclean.values[~np.isnan(adjrand * cclean.values)])

        corrCMMean[0, k - 1] = np.nanmean(corrCM)
        corrCMStd[0, k - 1] = np.nanstd(corrCM)
        distancesCM[0, k - 1] = np.nanmean(distCM)
        densitiesCM[0, k - 1] = np.nanmean(densityCM)

        corrBMMean[0, k - 1] = np.nanmean(corrBM)
        corrBMStd[0, k - 1] = np.nanstd(corrBM)
        distancesBM[0, k - 1] = np.nanmean(distBM)
        densitiesBM[0, k - 1] = np.nanmean(densityBM)

        corrERMean[0, k - 1] = np.nanmean(corrER)
        corrERStd[0, k - 1] = np.nanstd(corrER)
        distancesER[0, k - 1] = np.nanmean(distER)
        densitiesER[0, k - 1] = np.nanmean(densityER)

        corrCMMean[1, k - 1] = np.nanmean(corrCM_c)
        corrCMStd[1, k - 1] = np.nanstd(corrCM_c)
        distancesCM[1, k - 1] = np.nanmean(distCM_c)
        densitiesCM[1, k - 1] = np.nanmean(densityCM_c)

        corrBMMean[1, k - 1] = np.nanmean(corrBM_c)
        corrBMStd[1, k - 1] = np.nanstd(corrBM_c)
        distancesBM[1, k - 1] = np.nanmean(distBM_c)
        densitiesBM[1, k - 1] = np.nanmean(densityBM_c)

        corrERMean[1, k - 1] = np.nanmean(corrER_c)
        corrERStd[1, k - 1] = np.nanstd(corrER_c)
        distancesER[1, k - 1] = np.nanmean(distER_c)
        densitiesER[1, k - 1] = np.nanmean(densityER_c)

    allcorrTr = np.array(allcorrTr).flatten()
    allcorrCM = np.array(allcorrCM).flatten()
    allcorrBM = np.array(allcorrBM).flatten()
    allcorrER = np.array(allcorrER).flatten()

    # Save results
    np.save(PATH_EXPORT + "corrCMMean.npy", corrConnected)

    np.save(PATH_EXPORT + "corrCMMean.npy", corrCMMean)
    np.save(PATH_EXPORT + "corrCMStd.npy", corrCMStd)
    np.save(PATH_EXPORT + "distancesCM.npy", distancesCM)
    np.save(PATH_EXPORT + "densitiesCM.npy", densitiesCM)

    np.save(PATH_EXPORT + "corrBMMean.npy", corrBMMean)
    np.save(PATH_EXPORT + "corrBMStd.npy", corrBMStd)
    np.save(PATH_EXPORT + "distancesBM.npy", distancesBM)
    np.save(PATH_EXPORT + "densitiesBM.npy", densitiesBM)

    np.save(PATH_EXPORT + "corrERMean.npy", corrERMean)
    np.save(PATH_EXPORT + "corrERStd.npy", corrERStd)
    np.save(PATH_EXPORT + "distancesER.npy", distancesER)
    np.save(PATH_EXPORT + "densitiesER.npy", densitiesER)

    np.save(PATH_EXPORT + "allcorrTr.npy", allcorrTr)
    np.save(PATH_EXPORT + "allcorrCM.npy", allcorrCM)
    np.save(PATH_EXPORT + "allcorrBM.npy", allcorrBM)
    np.save(PATH_EXPORT + "allcorrER.npy", allcorrER)
