import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINEConv, global_add_pool, SAGPooling


class GinLayer(torch.nn.Module):
    def __init__(self, node_dim, out_dim, edge_dim):
        super(GinLayer, self).__init__()
        x_nn = Sequential(Linear(node_dim, out_dim),
                          ReLU(),
                          Linear(out_dim, out_dim))
        self.x_conv = GINEConv(nn=x_nn,
                               edge_dim=edge_dim)
        self.x_bn = torch.nn.BatchNorm1d(out_dim)
        self.e_bn = torch.nn.BatchNorm1d(out_dim)
        self.e_linear = Sequential(Linear(edge_dim + 2 * out_dim, out_dim),
                                   ReLU(),
                                   Linear(out_dim, out_dim),
                                   ReLU(),
                                   Linear(out_dim, out_dim))
        self.dim = out_dim

    def forward(self, em, edge_index, edge_features):
        # update node
        x_em = self.x_conv(x=em,
                           edge_index=edge_index,
                           edge_attr=edge_features)
        x_em = F.relu(self.x_bn(x_em))
        # update edge
        vij = x_em[edge_index].transpose(0, 1).reshape(-1, self.dim * 2)
        edge_mess = torch.cat((vij, edge_features), dim=1)
        edge_features = F.relu(self.e_bn(self.e_linear(edge_mess)))
        return x_em, edge_features


class GinBlocks(torch.nn.Module):
    def __init__(self, device, layers, dim):
        super(GinBlocks, self).__init__()
        self.embedding1 = torch.nn.Embedding(20, 10).to(device)
        self.embedding2 = torch.nn.Embedding(4, 2).to(device)
        self.edge_embedding1 = torch.nn.Embedding(2, 7).to(device)

        self.layers = layers
        self.num_features = 14
        self.e_num_features = 8
        self.dim = dim
        self.device = device
        first_layer = torch.nn.ModuleList([
            GinLayer(node_dim=self.num_features,
                     out_dim=dim,
                     edge_dim=self.e_num_features)
        ])
        other_layer = torch.nn.ModuleList([
            GinLayer(node_dim=dim,
                     out_dim=dim,
                     edge_dim=dim)
            for _ in range(layers - 1)
        ])
        self.encoder_layers = first_layer + other_layer

    def forward(self, x, edge_index, batch, pos, edge_features):
        x_1 = self.embedding1(x[:, 0].long())   # res_type
        x_2 = x[:, 1].unsqueeze(-1) / 100       # sasa
        x_3 = pos / 10       # position(with trans)
        x_em = torch.cat((x_1, x_2, x_3), dim=1)

        e_1 = self.edge_embedding1(edge_features[:, 0].long())
        e_2 = edge_features[:, 1].unsqueeze(-1) / 10  # dis
        edge_features_temp = torch.cat((e_1, e_2), dim=1)
        edge_features_em = torch.zeros(edge_features_temp.shape).to(self.device)
        # print(x_em.shape)
        # print(edge_features_em.shape)

        for layer in self.encoder_layers:
            x_em, edge_features_em = layer(em=x_em,
                                           edge_index=edge_index,
                                           edge_features=edge_features_em)
        out = global_add_pool(edge_features_em, batch)
        return out


class BA(torch.nn.Module):
    def __init__(self, device, layers):
        super(BA, self).__init__()
        self.GINmodel = GinBlocks(device=device,
                                  layers=layers,
                                  dim=128)
        self.linear = Sequential(Linear(256, 256),
                                 ReLU(),
                                 Linear(256, 256),
                                 ReLU(),
                                 Linear(256, 1))

    def forward(self, data):
        graph1 = self.GINmodel(x=data.x_s,
                               edge_index=data.edge_index_s,
                               batch=data.edge_features_s_batch,
                               pos=data.pos_s,
                               edge_features=data.edge_features_s)
        graph2 = self.GINmodel(x=data.x_t,
                               edge_index=data.edge_index_t,
                               batch=data.edge_features_t_batch,
                               pos=data.pos_t,
                               edge_features=data.edge_features_t)
        x1 = torch.cat((graph1, graph2), dim=-1)
        x2 = torch.cat((graph2, graph1), dim=-1)
        x = x1 + x2
        x = self.linear(x)
        return x