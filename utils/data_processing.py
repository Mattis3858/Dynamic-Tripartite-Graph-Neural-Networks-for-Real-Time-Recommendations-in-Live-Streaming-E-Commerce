import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def _load_tripartite(dataset_name, base_dir="training_data"):
    graph_df = pd.read_csv(f"{base_dir}/ml_{dataset_name}_edges.csv")
    edge_features = np.load(f"{base_dir}/ml_{dataset_name}_features.npy")
    node_features = np.load(f"{base_dir}/ml_{dataset_name}_node.npy")

    # 欄位期望：src, dst, ts, label, idx
    required_cols = {"src", "dst", "ts", "label", "idx"}
    missing = required_cols - set(graph_df.columns)
    if missing:
        raise ValueError(f"Missing columns in edges csv: {missing}")

    return graph_df, node_features, edge_features


def get_data_node_classification(
    dataset_name, use_validation=False, base_dir="training_data"
):
    random.seed(2020)
    graph_df, node_features, edge_features = _load_tripartite(
        dataset_name, base_dir=base_dir
    )

    # val_time, test_time = list(np.quantile(graph_df.ts.values, [0.70, 0.85]))

    sources = graph_df.src.values
    destinations = graph_df.dst.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    
    order = np.argsort(graph_df.ts.values, kind="mergesort")  # 稳定排序
    n = len(order)
    train_end = max(1, int(0.70 * n))
    val_end   = max(train_end + 1, int(0.85 * n))  # 確保 val 不為空；test 至少留 1 筆

    idx_train = order[:train_end]
    idx_val   = order[train_end:val_end]
    idx_test  = order[val_end:]

    mask = np.zeros(n, dtype=bool)
    train_mask = mask.copy(); train_mask[idx_train] = True
    val_mask   = mask.copy(); val_mask[idx_val]   = True
    test_mask  = mask.copy(); test_mask[idx_test] = True


    # train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    # test_mask = timestamps > test_time
    # val_mask = (
    #     np.logical_and(timestamps <= test_time, timestamps > val_time)
    #     if use_validation
    #     else test_mask
    # )

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )
    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )
    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(
    dataset_name,
    different_new_nodes_between_val_and_test=False,
    randomize_features=False,
    base_dir="training_data",
):
    graph_df, node_features, edge_features = _load_tripartite(
        dataset_name, base_dir=base_dir
    )

    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    val_time, test_time = list(np.quantile(graph_df.ts.values, [0.70, 0.85]))

    sources = graph_df.src.values
    destinations = graph_df.dst.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # 篩出 val 之後會出現的節點做為可抽樣母體
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time])
    )
    test_node_list = sorted(test_node_set)
    # k 取 10% 的總唯一節點，並加上 guard
    k = int(0.1 * n_total_unique_nodes)
    if k <= 0:
        k = 1
    k = min(k, len(test_node_list))  # 不能超過母體
    new_test_node_set = set(random.sample(test_node_list, k)) if k > 0 else set()

    # 將 new_test_node_set 當成「只在 val/test 期才會見到的新節點」
    # 注意：下方用 src/dst（不是 u/i）
    new_test_source_mask = graph_df.src.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.dst.map(
        lambda x: x in new_test_node_set
    ).values
    observed_edges_mask = np.logical_and(
        ~new_test_source_mask, ~new_test_destination_mask
    )

    # 訓練集：時間 <= val_time 且不包含新節點
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    # 定義 new_node_set：未出現在訓練集的節點
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_list = sorted(list(new_test_node_set))[:n_new_nodes]
        test_new_node_list = sorted(list(new_test_node_set))[n_new_nodes:]
        val_new_node_set = set(val_new_node_list)
        test_new_node_set = set(test_new_node_list)

        edge_contains_new_val_node_mask = np.array(
            [
                (a in val_new_node_set or b in val_new_node_set)
                for a, b in zip(sources, destinations)
            ]
        )
        edge_contains_new_test_node_mask = np.array(
            [
                (a in test_new_node_set or b in test_new_node_set)
                for a, b in zip(sources, destinations)
            ]
        )
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array(
            [
                (a in new_node_set or b in new_node_set)
                for a, b in zip(sources, destinations)
            ]
        )
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )
    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    new_node_val_data = Data(
        sources[new_node_val_mask],
        destinations[new_node_val_mask],
        timestamps[new_node_val_mask],
        edge_idxs[new_node_val_mask],
        labels[new_node_val_mask],
    )
    new_node_test_data = Data(
        sources[new_node_test_mask],
        destinations[new_node_test_mask],
        timestamps[new_node_test_mask],
        edge_idxs[new_node_test_mask],
        labels[new_node_test_mask],
    )

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.n_interactions, full_data.n_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes
        )
    )
    print(
        "The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes
        )
    )
    print(
        "The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes
        )
    )
    print(
        "{} nodes were used for the inductive testing, i.e. are never seen during training".format(
            len(new_test_node_set)
        )
    )

    return (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    )


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return (
        mean_time_shift_src,
        std_time_shift_src,
        mean_time_shift_dst,
        std_time_shift_dst,
    )
