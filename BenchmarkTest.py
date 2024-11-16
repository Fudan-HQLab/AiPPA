import numpy as np
import torch
from matplotlib import pyplot as plt, rcParams
from matplotlib.ticker import MultipleLocator
from scipy import stats
from torch_geometric.loader import DataLoader
from dataset import GNNDataset
from model import BA


def test(model, dataset, device):
    # calculate R (correlation)
    with torch.no_grad():
        model.eval()
        y_ = torch.tensor([])
        pred_ = torch.tensor([])

        name = []
        for batch in dataset:
            y = batch.y.to(device)
            # batch = batch.to(self.device)
            batch = batch.to(device)
            pred = model(data=batch)

            y_ = torch.cat((y_, torch.flatten(y.to('cpu'))), dim=-1)
            pred_ = torch.cat((pred_, torch.flatten(pred.to('cpu'))), dim=-1)
            name += batch.name
        r, p = stats.pearsonr(pred_, y_)
        if p > 0.05:
            r = 0.0
        whole_loss = torch.nn.MSELoss()(pred_, y_)
    print(name)
    return whole_loss, round(r, 3), p, y_, pred_


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load pre_train_models
model = BA(device=device, layers=2)
model = model.to(device)

model.load_state_dict(torch.load('model/model_trained.pth'))

test_dataset = GNNDataset(root='features/pkls/edge_thr_8',
                          phase='test')

test_loader = DataLoader(test_dataset,
                         batch_size=64,
                         num_workers=6,
                         shuffle=False,
                         follow_batch=['edge_features_s', 'edge_features_t'])
loss, r, p, ex_ba, pre_ba = test(model, test_loader, device)
print(f'MSE:{loss}')
print(f'R:{r}')
print(f'p:{p}')

config = {
    "font.family": 'Nimbus Sans',
    "font.size": 8,
    "font.style": 'normal',
    "font.weight": 'normal',
    "xtick.direction": "in",  # x轴刻度向内
    "ytick.direction": "in"  # y轴刻度向内

}
rcParams.update(config)

plt.scatter(ex_ba, pre_ba, c='#19CAAD', zorder=3, s=10)
plt.text(-10, -17, f'R={r:.2f}', fontsize=7)
# plt.text(-15, -3, f'p: {p:.2e}')
plt.text(-10, -18.3, f'RMSE={np.sqrt(loss):.2f} kcal/mol', fontsize=7)
x = np.linspace(-21, 1, 2000)
plt.plot(x, x, linewidth=0.5, color='orange')

plt.text(-17, -4, f'N={len(test_dataset)}')

plt.xlabel('Experimental binding affinity (kcal/mol)', labelpad=0.5, fontsize=8)
plt.ylabel('Predicted binding affinity (kcal/mol)', labelpad=0.5, fontsize=8)
plt.title('Performance on the test dataset', fontsize=8)
plt.tick_params(labelsize=7)
plt.tick_params(pad=0.7)
ax = plt.gca()
plt.xlim(-21, 1)
plt.ylim(-21, 1)

yminorLocator = MultipleLocator(1)  # 根据主刻度值，仔细设置次刻度标签为1的倍数
ax.yaxis.set_minor_locator(yminorLocator)
ymajorLocator = MultipleLocator(5)  # 根据主刻度值，仔细设置次刻度标签为1的倍数
ax.yaxis.set_major_locator(ymajorLocator)

xminorLocator = MultipleLocator(1)  # 根据主刻度值，仔细设置次刻度标签为1的倍数
ax.xaxis.set_minor_locator(xminorLocator)
xmajorLocator = MultipleLocator(5)  # 根据主刻度值，仔细设置次刻度标签为1的倍数
ax.xaxis.set_major_locator(xmajorLocator)

fig = plt.gcf()
fig.set_size_inches(2.95, 2.95)
plt.savefig('test_performance.png', dpi=300)
# plt.show()
