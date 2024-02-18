import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.spatial.distance import euclidean, cityblock
from sklearn.base import clone
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)
from IPython.display import HTML
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from IPython.display import display, Markdown
import plotly.offline as pyo

def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each
        corresponding to the coordinates of each point

    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    Wk = []
    for group in set(y):
        sum1 = 0
        n = 0
        for index, num in enumerate(y):
            if num == group:
                # Calculates the sum of the squares distance per cluster
                sum1 += (dist(X[index], centroids[num]) ** 2)

                # Counts the number of point/s in a cluster
                n += 1
        Wk.append(sum1 / (2 * n))

    return round(sum(Wk), 2)


def gen_realizations(X, b, random_state=None):
    """Generate b random realizations of X

    The realizations are drawn from a uniform distribution over the range of
    observed values for that feature.

    Parameters
    ---------
    X : array
        Design matrix with each row corresponding to a point
    b : int
        Number of realizations for the reference distribution
    random_state : int, default=None
        Determines random number generation for realizations

    Returns
    -------
    X_realizations : array
        random realizations with shape (b, X.shape[0], X.shape[1])
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = np.random.default_rng(random_state)
    nrows, ncols = X.shape
    return rng.uniform(
        np.tile(mins, (b, nrows, 1)),
        np.tile(maxs, (b, nrows, 1)),
        size=(b, nrows, ncols),
    )


def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a data point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference
        realizations
    random_state : int, default=None
        Determines random number generation for realizations

    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    X_refs = gen_realizations(X, b, random_state)

    Wki_values = []
    Wk = pooled_within_ssd(X, y, centroids, dist)

    for realization in X_refs:
        # cluster the generated data based on the input
        clusterer.fit(realization)
        labels = clusterer.labels_
        centroids = clusterer.cluster_centers_

        Wki = pooled_within_ssd(realization, labels, centroids, dist)
        Wki_values.append(Wki)

    gap_stat = []
    for num in Wki_values:
        gap_stat.append(np.log(num) - np.log(Wk))

    return [np.mean(gap_stat), np.std(gap_stat)]


def purity(y_true, y_pred):
    """Compute the class purity

    Parameters
    ----------
    y_true : array
        List of ground-truth labels
    y_pred : array
        Cluster labels

    Returns
    -------
    purity : float
        Class purity
    """
    # Create a confusion matrix
    conf_m = confusion_matrix(y_true, y_pred)

    purity = np.sum(np.max(conf_m, axis=0)) / np.sum(conf_m)

    return purity

def cluster_range(X, clusterer, k_start, k_stop):
    """Cluster X for different values of k

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a data point
    clusterer : sklearn.base.ClusterMixin
        Perform clustering for different value of `k` using this model. It
        should have been initialized with the desired parameters
    k_start : int
        Perform k-means starting from this value of `k`
    k_stop : int
        Perform k-means up to this value of `k` (inclusive)

    Returns
    -------
    dict
        The value of each key is a list with elements corresponding to the
        value at `k`. The keys are:
            * `ys`: cluster labels
            * `centers`: cluster centroids
            * `inertias`: sum of squared distances to the cluster centroid
            * `chs`: Calinski-Harabasz indices
            * `scs`: silhouette coefficients
            * `dbs`: Davis-Bouldin indices
            * `gss`: gap statistics
            * `gssds`: standard deviations of gap statistics
    """
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []
    for k in range(k_start, k_stop + 1):
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k)
        clusterer_k.fit(X)
        y = clusterer_k.labels_

        gs = gap_statistic(
            X,
            y,
            clusterer_k.cluster_centers_,
            euclidean,
            5,
            clone(clusterer).set_params(n_clusters=k),
            random_state=1337,
        )
        gss.append(gs[0])
        gssds.append(gs[1])
        ys.append(y)
        centers.append(clusterer_k.cluster_centers_)
        inertias.append(clusterer_k.inertia_)
        chs.append(calinski_harabasz_score(X, y))
        scs.append(silhouette_score(X, y))
        dbs.append(davies_bouldin_score(X, y))
    return {"centers": centers, "chs": chs, "dbs": dbs,
            "gss": gss, "gssds": gssds, "inertias": inertias,
            "scs": scs, "ys": ys}


def plot_clusters(X, ys, centers, transformer, figsize, dpi=150, squeeze=False):
    """Plot clusters given the design matrix and cluster labels"""
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, squeeze=squeeze, dpi=dpi, sharex=True, sharey=True,
                           figsize=figsize, subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y,cs in zip(range(2, k_max+1), ys, centers):
        centroids_new = transformer.transform(cs)
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y)) + 1),
                marker='s',
                ec='k',
                lw=1
            )
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y))+1),
                marker='s',
                ec='k',
                lw=1
            )
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax


def plot_clusters2(
    combs, X, ys, centers, transformer, figsizes, dpi=150, squeeze=False, sharex=True, sharey=True,
):
    """Plot clusters given the design matrix and cluster labels"""
    for comb, figsize in zip(combs, figsizes):
        display(Markdown(f"##### PC{comb[0]}-PC{comb[1]} view"))
        k_max = len(ys) + 1
        k_mid = k_max//2 + 2
        fig, ax = plt.subplots(2, k_max//2, squeeze=squeeze, dpi=dpi, sharex=sharex, sharey=sharey,
                               figsize=figsize, subplot_kw=dict(adjustable='box'),
                               gridspec_kw=dict(wspace=0.01))
        for k,y,cs in zip(range(2, k_max+1), ys, centers):
            centroids_new = transformer.transform(cs)
            if k < k_mid:
                ax[0][k%k_mid-2].scatter(X[:, comb[0]-1],  X[:, comb[1]-1], c=y, s=1, alpha=0.8)
                ax[0][k%k_mid-2].scatter(
                    centroids_new[:,comb[0]-1],
                    centroids_new[:,comb[1]-1],
                    s=10,
                    c=range(int(max(y)) + 1),
                    marker='s',
                    ec='k',
                    lw=1
                )
                ax[0][k%k_mid-2].set_title('$k=%d$'%k)
            else:
                ax[1][k%k_mid].scatter(X[:, comb[0]-1],  X[:, comb[1]-1], c=y, s=1, alpha=0.8)
                ax[1][k%k_mid].scatter(
                    centroids_new[:,comb[0]-1],
                    centroids_new[:,comb[1]-1],
                    s=10,
                    c=range(int(max(y))+1),
                    marker='s',
                    ec='k',
                    lw=1
                )
                ax[1][k%k_mid].set_title('$k=%d$'%k)
        plt.show()
    # return ax


def plot1(Z):
    """Plot the dendogram"""
    fig, ax = plt.subplots()
    dendrogram(Z, p=5, truncate_mode='level', get_leaves=True, ax=ax)
    ax.set_title(f"Dendogram for ")
    return ax


def plot12(Z, type_):
    """Plot the dendogram"""
    fig, ax = plt.subplots()
    dendrogram(Z, p=5, truncate_mode='level', get_leaves=True, ax=ax)
    ax.set_title(f"Dendogram for {type_} linkage")
    return ax


def plot_internal(ax, inertias, chs, scs, dbs, gss, gssds):
    """Plot internal validation values"""
    ks = np.arange(2, len(inertias) + 2)
    ax.plot(ks, inertias, "-o", label="SSE")
    ax.plot(ks, chs, "-ro", label="CH")
    ax.set_xlabel("$k$")
    ax.set_ylabel("SSE/CH")
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt="-go", label="Gap statistic")
    ax2.plot(ks, scs, "-ko", label="Silhouette coefficient")
    ax2.plot(ks, dbs, "-gs", label="DB")
    ax2.set_ylabel("Gap statistic/Silhouette/DB")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))
    # ax2.legend(lines + lines2, labels + labels2, )
    return ax

def plot_internal2(ax, inertias, chs, scs, dbs, gss, gssds, plot_dict):
    """Plot internal validation values"""
    ks = np.arange(2, len(inertias) + 2)
    if plot_dict.get('sse'):
        ax.plot(ks, inertias, "-o", label="SSE")
    if plot_dict.get('ch'):
        ax.plot(ks, chs, "-ro", label="CH")
    ax.set_xlabel("$k$")
    ax.set_ylabel("SSE/CH")
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    if plot_dict.get('gap'):
        ax2.errorbar(ks, gss, gssds, fmt="-go", label="Gap statistic")
    if plot_dict.get('sc'):
        ax2.plot(ks, scs, "-ko", label="Silhouette coefficient")
    if plot_dict.get('db'):
        ax2.plot(ks, dbs, "-gs", label="DB")
    ax2.set_ylabel("Gap statistic/Silhouette/DB")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))
    # ax2.legend(lines + lines2, labels + labels2, )
    return ax


def plot_internal_ch(ax, inertias, chs,):
    """Plot internal validation values"""
    ks = np.arange(2, len(inertias) + 2)
    ax.plot(ks, chs, "-ro", label="CH")
    ax.set_xlabel("$k$")
    ax.set_ylabel("SSE/CH")
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))
    return ax


def get_kdist(k, data):
    """Get nearest neigbors"""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    k_nearest_distances = distances[:, -1]
    k_nearest_distances = sorted(k_nearest_distances, reverse=True)

    return k_nearest_distances


def heirarch_cluster(combs, type_, dist_, df, df_reduced):
    agg = AgglomerativeClustering(
        n_clusters=None, linkage=type_, distance_threshold=dist_
    )
    cluster_labels = agg.fit_predict(df)

    fig, axs = plt.subplots(nrows=1, ncols=len(combs), figsize=(12, 4))
    for i, comb in enumerate(combs):
        ax = axs[i] if len(combs) > 1 else axs
        ax.scatter(
            df_reduced[:, comb[0] - 1], df_reduced[:, comb[1] - 1],
            c=cluster_labels
        )
        ax.set_title(f"Scatter Plot of {type_} Linkage PC{comb[0]}-PC{comb[1]} view")
        ax.set_xlabel(f"PC {comb[0]}")
        ax.set_ylabel(f"PC {comb[1]}")

    plt.tight_layout()
    plt.show()

    Z = linkage(df, method=type_, optimal_ordering=True)
    plot12(Z, type_)
    plt.show()

    return cluster_labels


def choro_incident_counts_gpds(df_cluster, df_boroughs, plot_list):
    df_b = df_boroughs.copy()
    choro_df = df_cluster['borough'].value_counts().to_frame().reset_index()
    choro_df['borough'] = choro_df['borough'].str.lower()

    for index, row in df_b.iterrows():
        for index_, row_ in choro_df.iterrows():
            if row_['borough'] in row['DISTRICT']:
                df_b.loc[index, 'count'] = row_['count']
    df_b.fillna(0)
    # display(df_b)

    # fig, ax = plt.subplots(1, figsize=(8, 8))
    plot_list.append(df_b)
    # df_b.plot(column='count', cmap='Reds', legend=True, ax=ax, edgecolor='black')
    # ax.axis('off')
    # # df_b.apply(
    # #     lambda x: ax.annotate(text=x['DISTRICT'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=7), axis=1)
    # fig.show()


def choro_incident_counts(df_cluster, local_auth_list, local_auth):
    choro_df = df_cluster['borough'].value_counts().reset_index()
    choro_df['borough'] = choro_df['borough'].map(
        lambda x: local_auth_list[[i.lower() for i in local_auth_list].index(x.lower())]
        if x.lower() in [i.lower() for i in local_auth_list] else None
    )
    choro_df.columns = ['LA', 'Val']

    for la in local_auth_list:
        if la not in choro_df['LA'].tolist():
            new_row = {'LA': la, 'Val': '0'}
            new_index = len(choro_df)
            choro_df.loc[new_index] = new_row

    # choropleth_mapbox
    fig = px.choropleth_mapbox(
        choro_df,
        geojson=local_auth,
        locations='LA',
        color='Val',
        featureidkey="properties.LAD21NM",
        color_continuous_scale="Inferno",
        mapbox_style="carto-positron",
        center={"lat": 51.4399, "lon": -0.0943},
        zoom=8,
        labels={'val': 'value'},
    )
    fig.update_layout(height=420, width=420, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    # import plotly.graph_objects as go
    #
    # fig = go.Figure(go.Choroplethmapbox(
    #     geojson=local_auth,
    #     locations=choro_df['LA'],
    #     z=choro_df['Val'],
    #     featureidkey="properties.LAD21NM",
    #     colorscale="Reds",
    #     colorbar=dict(title='Value'),
    #     marker_line_width=0,
    # ))
    #
    # fig.update_layout(
    #     height=420, width=420,
    #     mapbox=dict(
    #         style="carto-positron",
    #         center=dict(lat=51.4399, lon=-0.0943),
    #         zoom=8,
    #     )
    # )
    # fig.show()


def choro_financial(sheet_name, london_boroughs):
    df = pd.read_excel('1998_2021_finance_UK.xlsx', sheet_name=sheet_name, skiprows=1)
    df = df.drop([str(i) for i in range(1998, 2021)], axis=1)
    lowercase_london_boroughs = [borough.lower() for borough in london_boroughs]
    df = df[df['LA name'].str.lower().isin(lowercase_london_boroughs)]
    df = df.drop(['ITL1 Region', 'LA code'], axis=1)

    with open('GB_local_authority.json', 'r') as f:
        local_auth = json.load(f)

    local_auth_list = []
    for i in range(len(local_auth["features"])):
        local_a = local_auth["features"][i]['properties']['LAD21NM']
        local_auth_list.append(local_a)

    df['LA name'] = df['LA name'].map(
        lambda x: local_auth_list[[i.lower() for i in local_auth_list].index(x.lower())]
        if x.lower() in [i.lower() for i in local_auth_list] else None
    )
    df.columns = ['LA', 'Val']

    fig = px.choropleth_mapbox(
        df,
        geojson=local_auth,
        locations='LA',
        color='Val',
        featureidkey="properties.LAD21NM",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        center={"lat": 51.5072, "lon": 0.1276},
        zoom=8.4,
        labels={'val':'value'},

    )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


def kmeans_proper(combs, pca, n_clusters, df, reduced_df):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1337, n_init='auto')
    y_predict = kmeans.fit_predict(df.values)

    fig, axs = plt.subplots(nrows=1, ncols=len(combs), figsize=(12, 4))

    for i, comb in enumerate(combs):
        ax = axs[i] if len(combs) > 1 else axs
        ax.scatter(reduced_df[:, comb[0] - 1], reduced_df[:, comb[1] - 1], c=y_predict)
        kmeans_new = pca.transform(kmeans.cluster_centers_)
        scatter = ax.scatter(
            kmeans_new[:, comb[0] - 1],
            kmeans_new[:, comb[1] - 1],
            s=60,
            c=range(kmeans.n_clusters),
            marker="s",
            ec="k",
            lw=2,
        )
        ax.set_title(f"Scatter Plot of K-Means PC{comb[0]}-PC{comb[1]} view")
        ax.set_xlabel(f"PC {comb[0]}")
        ax.set_ylabel(f"PC {comb[1]}")

        legend1 = ax.legend(
            *scatter.legend_elements(),
            title="Cluster Labels",
            loc="upper right"
        )
        ax.add_artist(legend1)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(*zip(*reduced_df), c=y_predict)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('K-Means Clustering in 3D')
    fig.colorbar(scatter, ax=ax, label='Cluster', pad=0.2)
    plt.show()

    return kmeans


def kmed_proper(combs, pca, n_clusters, df, reduced_df):
    kmed = KMedoids(n_clusters=n_clusters,  method="pam", random_state=1337)
    y_predict = kmed.fit_predict(df.values)

    fig, axs = plt.subplots(nrows=1, ncols=len(combs), figsize=(12, 3))

    for i, comb in enumerate(combs):
        ax = axs[i] if len(combs) > 1 else axs
        ax.scatter(reduced_df[:, comb[0] - 1], reduced_df[:, comb[1] - 1], c=y_predict)
        kmeans_new = pca.transform(kmed.cluster_centers_)
        scatter = ax.scatter(
            kmeans_new[:, comb[0] - 1],
            kmeans_new[:, comb[1] - 1],
            s=60,
            c=range(kmed.n_clusters),
            marker="s",
            ec="k",
            lw=2,
        )
        ax.set_title(f"Scatter Plot of K-Medoids PC{comb[0]}-PC{comb[1]} view")
        ax.set_xlabel(f"PC {comb[0]}")
        ax.set_ylabel(f"PC {comb[1]}")

        legend1 = ax.legend(
            *scatter.legend_elements(),
            title="Cluster Labels",
            loc="upper right"
        )
        ax.add_artist(legend1)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(*zip(*reduced_df), c=y_predict)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('K-Medoids Clustering in 3D')
    fig.colorbar(scatter, ax=ax, label='Cluster', pad=0.2)
    plt.show()

    return kmed


def print_cluster_metrics(
    n_cluster, to_label, local_auth_list, local_auth, cluster_type, stats_summary_list,
    cat_summary_list, use_gpds, df_boroughs, plot_list, cluster_metrics_to_investigate
):
    # print(125*"=")
    # print(f"cluster {n_cluster}")
    df_cluster = to_label[to_label[cluster_type] == n_cluster]
    # display(df_cluster.info()

    stats_df = df_cluster[
        cluster_metrics_to_investigate
    ].describe()
    stats_df.drop(['25%', '50%', '75%'], axis=0, inplace=True)
    stats_df.index = pd.MultiIndex.from_product([[f"Cluster {n_cluster}"], stats_df.index])
    stats_summary_list.append(stats_df)

    # display(stats_df)
    dpc = df_cluster[['PropertyCategory']].value_counts().to_frame('counts').reset_index().head(5)
    dpt = df_cluster['PropertyType'].value_counts().to_frame('counts').reset_index().head(5)
    db = df_cluster['borough'].value_counts().to_frame('counts').reset_index().head(5)
    cat_df = pd.concat([dpc, dpt, db], axis=1)
    cat_df.index = pd.MultiIndex.from_product([[f"Cluster {n_cluster}"], cat_df.index])
    cat_summary_list.append(cat_df)

    if use_gpds:
        choro_incident_counts_gpds(df_cluster, df_boroughs, plot_list)
    else:
        choro_incident_counts(df_cluster, local_auth_list, local_auth)

    first_from_main_station = len(df_cluster[df_cluster['IncidentStationGround'] == df_cluster['FirstPumpArriving_DeployedFromStation']]) \
        / df_cluster['FirstPumpArriving_DeployedFromStation'].count()
    second_from_main_station = len(df_cluster[df_cluster['IncidentStationGround'] == df_cluster['SecondPumpArriving_DeployedFromStation']]) \
        / df_cluster['SecondPumpArriving_DeployedFromStation'].count()
    # print(f"\nFirst percent coming from main station {first_from_main_station*100:.2f}%")
    # print(f"Second percent coming from main station {second_from_main_station*100:.2f}%")
    # print(125*"=")


def prep_data_for_clustering(conn, filter_, suffix, save):
    if filter_:
        add_q = filter_
    else:
        add_q = ""

    query = f"""
    SELECT DISTINCT
        IncidentNumber,
        IncidentGroup,
        CalYear CallYear,
        CAST(SUBSTR(DateOfCall, 6, 2) AS INTEGER) CallMonth,
        CAST(SUBSTR(DateOfCall, 9, 2) AS INTEGER) AS CallDay,
        HourOfCall HourOfCall,
        strftime('%Y-%m-%d', DateOfCall) DateOfCall,
        strftime('%H:%M:%S', TimeOfCall) TimeOfCall,
        IncGeo_BoroughName as borough,
        IncGeo_WardName as ward,
        IncidentStationGround,
        FirstPumpArriving_DeployedFromStation,
        SecondPumpArriving_DeployedFromStation,
        PropertyCategory,
        PropertyType,
        FirstPumpArriving_AttendanceTime as first_pump_time,
        SecondPumpArriving_AttendanceTime as second_pump_time,
        NumStationsWithPumpsAttending as num_of_station_pumps,
        NumPumpsAttending as num_pumps,
        PumpCount as pump_cnt,
        PumpHoursRoundUp as pump_hrs_rnd_up,
        "Notional Cost (£)" as notional_cost,
        NumCalls as num_calls
    FROM incidents_202001_202308
    WHERE IncidentGroup = 'Fire'
    {add_q}
    """
    df_fire = pd.read_sql(query, conn).set_index('IncidentNumber')
    df_fire_orig = df_fire.copy()
    if save:
        df_fire.to_csv(f'tolabel_{suffix}.csv')
    df_fire = df_fire.fillna(0)

    numeric_cols = df_fire.select_dtypes(include='number')
    numeric_cols.drop(columns=['CallYear', 'CallMonth', 'CallDay', 'HourOfCall'], inplace=True)

    standard_scaler = StandardScaler()
    numeric_cols_scaled = standard_scaler.fit_transform(numeric_cols.to_numpy())
    df_prep = pd.DataFrame(numeric_cols_scaled, index=numeric_cols.index, columns=list(numeric_cols.columns))

    print(f"{suffix}: ")
    display(df_prep)
    if save:
        df_prep.to_csv(f'prep_{suffix}.csv')

    return df_prep, df_fire_orig


def display_pca_views(combs, df_reduced, cluster_labels):
    fig, axs = plt.subplots(nrows=1, ncols=len(combs), figsize=(12, 4))

    for i, comb in enumerate(combs):
        ax = axs[i] if len(combs) > 1 else axs
        ax.scatter(
            df_reduced[:, comb[0] - 1], df_reduced[:, comb[1] - 1],
            c=cluster_labels
        )
        ax.set_title(f"PC{comb[0]}-PC{comb[1]} view")
        ax.set_xlabel(f"PC {comb[0]}")
        ax.set_ylabel(f"PC {comb[1]}")

    plt.tight_layout()
    plt.show()


def format_with_commas(x):
    return '{:,.2f}'.format(x)


def interactive_ivc(wid, res):
    wid_ivc_dict = {}
    for i in wid:
        wid_ivc_dict.update({i: "show"})
    fig, ax = plt.subplots()
    plot_internal2(
        ax,
        res["inertias"],
        res["chs"],
        res["scs"],
        res["dbs"],
        res["gss"],
        res["gssds"],
        wid_ivc_dict
    )
    fig.show()
