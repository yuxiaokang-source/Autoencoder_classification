import torch
import ipdb
torch.manual_seed(1)#这个可以控制权重的初始化
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])


model = Model()#
print(model.state_dict())
print(model.layer.weight.data)
print(model.layer.bias.data)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(10):
    #ipdb.set_trace()
    optimizer.zero_grad()
    pred = model(x).squeeze()
    squared_diff = (y - pred) ** 2
    loss = squared_diff.mean()
    print(loss.item())
    loss.backward()
    print("==========weight更新前===============")
    print(model.layer.weight.data)
    print(model.layer.weight.data[0][0]-0.1*1/2*((pred.data[2]-4)+(pred.data[3]-6)))#W[0][0]
    optimizer.step()
    print("==========weight更新后===============")
    print(model.layer.weight.data)
    
#    OrderedDict([('layer.weight', tensor([[ 0.3643, -0.3121]])), ('layer.bias', tensor([-0.1371]))])
#tensor([[ 0.3643, -0.3121]])
#tensor([-0.1371])
#22.698406219482422
#==========weight更新前===============
#tensor([[ 0.3643, -0.3121]])
#tensor(0.8572)
#==========weight更新后===============
#tensor([[0.8572, 0.2646]])
#10.705572128295898
#==========weight更新前===============
#tensor([[0.8572, 0.2646]])
#tensor(1.1798)
#==========weight更新后===============
#tensor([[1.1798, 0.6668]])
#5.121640205383301
#==========weight更新前===============
#tensor([[1.1798, 0.6668]])
#tensor(1.3869)
#==========weight更新后===============
#tensor([[1.3869, 0.9495]])
#2.514791965484619
#==========weight更新前===============
#tensor([[1.3869, 0.9495]])
#tensor(1.5159)
#==========weight更新后===============
#tensor([[1.5159, 1.1504]])
#1.2915303707122803
#==========weight更新前===============
#tensor([[1.5159, 1.1504]])
#tensor(1.5923)
#==========weight更新后===============
#tensor([[1.5923, 1.2951]])
#0.7118869423866272
#==========weight更新前===============
#tensor([[1.5923, 1.2951]])
#tensor(1.6334)
#==========weight更新后===============
#tensor([[1.6334, 1.4010]])
#0.4321833848953247
#==========weight更新前===============
#tensor([[1.6334, 1.4010]])
#tensor(1.6509)
#==========weight更新后===============
#tensor([[1.6509, 1.4802]])
#0.29274702072143555
#==========weight更新前===============
#tensor([[1.6509, 1.4802]])
#tensor(1.6529)
#==========weight更新后===============
#tensor([[1.6529, 1.5407]])
#0.2193479686975479
#==========weight更新前===============
#tensor([[1.6529, 1.5407]])
#tensor(1.6448)
#==========weight更新后===============
#tensor([[1.6448, 1.5882]])
#0.17743359506130219
#==========weight更新前===============
#tensor([[1.6448, 1.5882]])
#tensor(1.6302)
#==========weight更新后===============
#tensor([[1.6302, 1.6264]])