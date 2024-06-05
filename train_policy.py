# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
import torch
import model
import dataset
from torch.utils.data import Dataset, DataLoader

global Time_Value, Batch_Size
Time_Value = 300
Batch_Size = 128


def train_policy():
    file = "./dataset/data_init.csv"
    log_file = open(r"param/log_policy.txt", "w")
    device = torch.device("cuda")
    data_set = dataset.ImitationData(file, demo_no=1, time_scale=Time_Value)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True)
    policy_model = model.PolicyNet(input=Time_Value).to(device)
    # mse loss
    criterion = torch.nn.MSELoss().to(device)
    # adam optimizer
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.000001)
    # train policy model
    policy_model.train()
    bc_models = []
    for i in range(5):
        bc_model = model.BehaviorCost(input=Time_Value)
        bc_model = bc_model.to(device)
        bc_model.load_state_dict(torch.load(r"param/bc_model_{}.pth".format(i)))
        bc_model.eval()
        bc_models.append(bc_model)
    for epoch in range(30):
        policy_stack = data_set.init_policy().to(device)
        err = 0
        for i, (_, _, _) in enumerate(data_loader):
            training_data = policy_stack
            # calculate ideal policy (maybe)
            with torch.set_grad_enabled(False):
                bc_stack = torch.cat(
                    [bc_model(training_data) for bc_model in bc_models]
                ).mean(dim=1)
                bc_stack = bc_stack.view(1, -1)

            with torch.set_grad_enabled(True):
                policy_new = policy_model(training_data)
                state_new = policy_new
                # calculate policy loss
                policy_loss = criterion(state_new, bc_stack)
                # backward propagation
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
                policy_new.detach_()
                state_new.detach_()
                # update policy stack
                policy_stack = torch.cat(
                    (policy_stack, policy_new.view(1, 1, -1)), dim=1
                )
                policy_stack = policy_stack[:, 1:, :]
                err = err * i / (i + 1) + policy_loss / (i + 1)
                if i % 1000 == 0:
                    print("[epoch:", epoch, ",i:", i, "],loss:", err.item())
                    log_file.write("[epoch:{},loss:{}]\n".format(epoch, err.item()))
                if i % 10000 == 0:
                    torch.save(
                        policy_model.state_dict(),
                        r"param/policy_temp/policy_model_{}_{}.pth".format(epoch, i),
                    )
        print("[epoch:", epoch, "],loss:", err.item())
        log_file.write("[epoch:{},loss:{}]\n".format(epoch, err.item()))
        torch.save(
            policy_model.state_dict(), r"param/policy_model_{}.pth".format(epoch)
        )
    log_file.close()


def generate_data():
    """
    generate data using policy model
    :return: write data into file 'policy.txt'
    """
    file = "./dataset/data_init.csv"
    device = torch.device("cuda")
    data_set = dataset.ImitationData(file, demo_no=1, time_scale=Time_Value)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True)
    policy_model = model.PolicyNet(input=Time_Value).to(device)
    policy_model.load_state_dict(torch.load(r"param/policy_model_29.pth"))
    policy_model.eval()
    policy_stack = data_set.init_policy().to(device)
    policy_record = policy_stack
    with torch.no_grad():
        for i, (_, _, _) in enumerate(data_loader):
            training_data = policy_stack
            # calculate ideal policy (maybe)
            policy_new = policy_model(training_data)
            policy_new = policy_new.view(1, -1)
            policy_stack = torch.cat((policy_stack, policy_new.view(1, 1, -1)), dim=1)
            policy_record = torch.cat((policy_record, policy_new.view(1, 1, -1)), dim=1)
            policy_stack = policy_stack[:, 1:, :]
            if i % 100 == 0:
                # print policy new
                print(policy_new)
    np.savetxt(
        "./demo/policy_test.csv",
        policy_record.view(-1, 5).cpu().detach().numpy(),
        fmt="%f",
        delimiter=",",
    )


if __name__ == "__main__":
    train_policy()
    generate_data()
