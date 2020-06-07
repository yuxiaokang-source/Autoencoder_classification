import torch
import torchsnooper
import snoop

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)
    
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])
#这里三个未知数，有四个点，理论上来说是有唯一正确解的。使用神经网络也是很好毕竟这些值的
print(y.size())
print(x.size())

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    optimizer.zero_grad()
    pred = model(x)
    #print(pred.size())#pred.size=[4,1]
    squared_diff = (y - pred) ** 2#可以使用调试工具，发现是这个地方有问题损失函数不收敛
    #print(squared_diff.size())
    loss = squared_diff.mean()
    print(loss.item())
    loss.backward()
    optimizer.step()#维度不对，结果损失不收敛
print(model.state_dict())  
print(model.layer.weight.data)
print(model.layer.bias.data)

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    optimizer.zero_grad()
    pred = model(x).squeeze()
    squared_diff = (y - pred) ** 2
    loss = squared_diff.mean()
    print(loss.item())
    loss.backward()
    optimizer.step()
parm={}
for name,parameters in model.named_parameters():
    print(name,':',parameters.size())
    parm[name]=parameters.detach().numpy()

#print(model.layer[0].weight.data)
print(model)
print(model.state_dict())
print(model.layer.weight.data)
print(model.layer.bias.data)