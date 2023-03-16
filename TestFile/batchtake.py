import torch

# a = torch.ones([64,3,88])
def get_one_agent_batch_data(BData):
    BData_per_agent = []
    for n in range(3):
        BData_per_agent.append([data[n] for data in BData])
    return BData_per_agent
a = [
        [[1,1,1,1],[2,1,1,1],[3,1,1,1]],
        [[1,1,1,1],[1,1,1,1],[1,1,1,1]],
        [[1,1,1,1],[1,1,1,1],[1,1,1,1]],
        [[1,1,1,1],[1,1,1,1],[1,1,1,1]]
    ]

d = get_one_agent_batch_data(a)
print(d[0])
print(d[1])
print(d[2])