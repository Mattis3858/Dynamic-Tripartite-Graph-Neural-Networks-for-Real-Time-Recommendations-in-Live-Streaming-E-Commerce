from utils.data_processing import get_data

node_feats, edge_feats, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    get_data("tripartite", different_new_nodes_between_val_and_test=False, randomize_features=False)

print("Node features shape:", node_feats.shape)
print("Edge features shape:", edge_feats.shape)
print("Full data: n_interactions =", full_data.n_interactions, ", n_unique_nodes =", full_data.n_unique_nodes)
print("Train data: n_interactions =", train_data.n_interactions, ", n_unique_nodes =", train_data.n_unique_nodes)
