# scripts/train_mini_lstm.py
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

data = np.load("data_proc/dataset_keypoints.npz", allow_pickle=True)
X=torch.tensor(data["X"],dtype=torch.float32)  # (N,32,111)
y=torch.tensor(data["y"],dtype=torch.long)
classes=list(data["classes"]); C=len(classes); F=X.shape[-1]

class TinyLSTM(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=64):
        super().__init__()
        self.lstm=nn.LSTM(in_dim,hidden,2,batch_first=True,bidirectional=True,dropout=0.1)
        self.fc=nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2,num_classes))
    def forward(self,x):
        y,_=self.lstm(x); y=y[:,-1,:]; return self.fc(y)

N=len(X); n_val=max(1,int(0.1*N)); n_test=max(1,int(0.1*N))
train,val,test = random_split(TensorDataset(X,y), [N-n_val-n_test,n_val,n_test],
                              generator=torch.Generator().manual_seed(42))
dl = DataLoader(train,batch_size=64,shuffle=True)
vl = DataLoader(val,batch_size=128); tl = DataLoader(test,batch_size=128)

dev="cuda" if torch.cuda.is_available() else "cpu"
m=TinyLSTM(F,C).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
crit=nn.CrossEntropyLoss(label_smoothing=0.05)

best=(0,None)
for e in range(20):
    m.train(); tloss=tacc=0
    for xb,yb in dl:
        xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad()
        lo=m(xb); loss=crit(lo,yb); loss.backward(); opt.step()
        tloss+=loss.item()*xb.size(0); tacc+=(lo.argmax(1)==yb).sum().item()
    tloss/=len(dl.dataset); tacc/=len(dl.dataset)
    m.eval(); vloss=vacc=0
    with torch.no_grad():
        for xb,yb in vl:
            xb,yb=xb.to(dev),yb.to(dev)
            lo=m(xb); vloss+=crit(lo,yb).item()*xb.size(0); vacc+=(lo.argmax(1)==yb).sum().item()
    vloss/=len(vl.dataset); vacc/=len(vl.dataset)
    print(f"Epoch {e+1:02d}  train {tloss:.3f}/{tacc:.3f}  val {vloss:.3f}/{vacc:.3f}")
    if vacc>best[0]: best=(vacc,m.state_dict())

torch.save({"model":best[1],"classes":classes,"feat_dim":F}, "models/mini_lstm_best.pth")
print("Saved models/mini_lstm_best.pth")
