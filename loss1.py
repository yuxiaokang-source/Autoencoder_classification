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
#��������δ֪�������ĸ��㣬��������˵����Ψһ��ȷ��ġ�ʹ��������Ҳ�ǺܺñϾ���Щֵ��
print(y.size())
print(x.size())

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    optimizer.zero_grad()
    pred = model(x)
    #print(pred.size())#pred.size=[4,1]
    squared_diff = (y - pred) ** 2#����ʹ�õ��Թ��ߣ�����������ط���������ʧ����������
    #print(squared_diff.size())
    loss = squared_diff.mean()
    print(loss.item())
    loss.backward()
    optimizer.step()#ά�Ȳ��ԣ������ʧ������
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