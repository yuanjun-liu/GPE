import numpy as np
import torch
import GPE

def node_gpe(dataset,dim):
    # load graph    
    dataPath = "./data/" + dataset
    nodePath = dataPath + "/road/node.csv"
    node_txt= open(nodePath,encoding='utf-8').read().replace('\r','').split('\n')[1:]
    node_pos={}
    for line in node_txt:
        line=line.replace('\n','')
        if ',' not in line:continue
        xs=line.split(',')
        nid,lon,lat=int(xs[0]),float(xs[1]),float(xs[2])
        node_pos[nid]=(lon,lat)
    df_node = pd.read_csv(nodePath, sep=',')
    num_node = df_node["node"].size
    print('num_node',num_node,len(node_pos))
    # gpe
    pos=torch.Tensor([node_pos[i] for i in range(num_node)])
    gpe=GPE(dim=dim)
    ebd=gpe(pos).cpu().numpy()
    np.save("./data/" + dataset + f"/node_gpe_{dim}.npy", ebd)

def node_gpe_merge(dataset):
    path_n2v_64= "./data/" + dataset + "/node_n2v_64.npy"
    path_gpe_64="./data/" + dataset + "/node_gpe_64.npy"
    path_gpe_n2v_128="./data/" + dataset + "/node_gpe_n2v_128.npy"
    n2v_64= np.load(path_n2v_64, allow_pickle=True)
    gpe_64= np.load(path_gpe_64, allow_pickle=True)
    gpe_n2v_128=np.concatenate([n2v_64,gpe_64],axis=-1)
    print('gpe_n2v_128 shape',gpe_n2v_128.shape)
    np.save(path_gpe_n2v_128, gpe_n2v_128)

if __name__ == "__main__":
    """ before patching
    1. run the original code, generate 64-dim node2vec embedding, and rename it to node_n2v_64.npy
    2. set the feature_size=128 in config.yaml
    3. run the original code, generate 128-dim node2vec embedding, and rename it to node_n2v_128.npy
    4. patch the code, generate the 128-dim gpe embedding of name node_gpe128.npy, and 128-dim gpe&node2vec embedding of name node_gpe_n2v_128.npy
    """
    node_gpe('tdrive',64)
    node_gpe('tdrive',128)
    node_gpe_merge('tdrive')
    exit(0)