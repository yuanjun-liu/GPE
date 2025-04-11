
# implement this function
def use_gpe():
    """return: 
    - n2v_128: the original node2vec embedding
    - gpe_128: the GPE embedding
    - gpe_n2v_128: combine the GPE with node2vec 
    """
    raise NotImplementedError('you need to implement use_gpe in data_utils')

# repalce this function
def load_netowrk(dataset):
    """
    load road network from file with Pytorch geometric data object
    :param dataset: the city name of road network
    :return: Pytorch geometric data object of the graph
    """
    edge_path = "./data/" + dataset + "/road/edge_weight.csv"

    if use_gpe()=='n2v_128':
        node_embedding_path = "./data/" + dataset + "/node_n2v_128.npy"
        print('load network node2vec')
    elif use_gpe()=='gpe_128':
        node_embedding_path = "./data/" + dataset + "/node_gpe_128.npy"
        print('load network gpe')
    elif use_gpe()=='gpe_n2v_128':
        node_embedding_path = "./data/" + dataset + "/node_gpe_n2v_128.npy"
        print('load network gpe + node2vec')
    else:raise RuntimeError('bad gpe in data_utils')

    node_embeddings = np.load(node_embedding_path)
    df_dege = pd.read_csv(edge_path, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_attr = df_dege["length"].to_numpy()

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    print("node embeddings shape: ", node_embeddings.shape)
    print("edge_index shap: ", edge_index.shape)
    print("edge_attr shape: ", edge_attr.shape)

    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network