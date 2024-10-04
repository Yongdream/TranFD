import torch
import numpy as np
import model_Conv
import model_Vit
import model_SPP
from Processor import dataset_get
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(network, train_loader, optimizer):
    # count = 0
    loss_list = []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):

        # data = data[:, 0, :, :].unsqueeze(1)  # 单通道对比实验（注意测试部分也需要对应删改）

        data_n = data.cpu()
        for m in range(0, data.shape[0]):
            current = np.array(data[m].permute(2, 1, 0).cpu()).astype(np.float32)
            np.random.shuffle(current)
            current = torch.from_numpy(current)
            data_n[m] = current.permute(2, 1, 0)
            data_n[m] = data_n[m].type(torch.FloatTensor)
        data_n = data_n.cuda()
        # data_n = data_n[:, 0, :, :].unsqueeze(1).cuda()

            # pict = Image.fromarray(np.uint8(255 * np.array(current.cpu())), mode="RGB")  # 调试
            # pict.show()

        target_n = np.zeros((target.shape[0], ))
        for m in range(0, target.shape[0]):
            for n in range(0, target.shape[2]):
                if target[m, 0, n] == 1:
                    target_n[m] = n
        target_n = torch.from_numpy(target_n).type(torch.FloatTensor).cuda()

        output = network(data_n)
        # print(output.shape)
        output = output.view(output.shape[0], -1)
        # print(output.shape)
        # target = target.view(target.shape[0], -1)
        loss = F.cross_entropy(output, target_n.long())
        # loss = F.nll_loss(output, target_n.long())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # grad = [x.grad for x in optimizer.param_groups[0]['params']]
        loss_list.append(float((loss/data.shape[0]).cpu()))

        optimizer.step()
    loss_ave = np.mean(np.array(loss_list), axis=0)
    print("loss_average:" + str(loss_ave))
    scheduler.step()
    return loss_ave
    # print(loss)
    #     count += 1
    # print(count)

def accuracy(epoch_idx, test_loader, network, set_type = None):
    correct = 0
    num_origin = np.zeros((5,))
    num_correct = np.zeros((5,))
    corr_list = np.zeros((5,))
    with torch.no_grad():
        for data, target in test_loader:

            # data = data[:, 0, :, :].unsqueeze(1)  # 单通道对比实验
            # data torch.Size([4, 3, 300, 20])

            target_n = np.zeros((target.shape[0],))
            for m in range(0, target.shape[0]):
                for n in range(0, target.shape[2]):
                    if target[m, 0, n] == 1:
                        target_n[m] = n

            outputs = network(data)         # torch.Size([4, 5])
            _, predicted = torch.max(outputs.squeeze(1).data, 1)
            # correct += (predicted == target).sum().item()
            for m in range(0, data.shape[0]):

                num_origin[int(target_n[m])] += 1

                max_index = predicted[m]
                index_m = np.zeros(target.shape[-1])
                index_m[max_index] = 1
                a = torch.from_numpy(index_m).unsqueeze(0).cuda()
                b = target[m]
                if torch.equal(a, b):   # 预测值和原始值是否一致
                    correct += 1
                    num_correct[int(target_n[m])] += 1
    for m in range(0, 5):
        corr_list[m] = 100. * num_correct[m] / num_origin[m]

    if set_type == "train":
        print('Epoch{}: Train accuracy: {}/{} ({:.2f}%)'.format(
            epoch_idx, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print("ACC:Cor:{:.2f}% ISC:{:.2f}% Noi:{:.2f}% Nor:{:.2f}% Vis :{:.2f}%".format(corr_list[0], corr_list[1],
                                                                                        corr_list[2], corr_list[3],
                                                                                        corr_list[4]))

    if set_type == "val":
        print('Epoch{}: Test accuracy: {}/{} ({:.2f}%)'.format(
            epoch_idx, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print("ACC:Cor:{:.2f}% ISC:{:.2f}% Noi:{:.2f}% Nor:{:.2f}% Vis :{:.2f}%".format(corr_list[0], corr_list[1],
                                                                                        corr_list[2], corr_list[3],
                                                                                        corr_list[4]))

    return correct / len(test_loader.dataset), corr_list

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with command line arguments')
    parser.add_argument('--select_condition', type=str, choices=['UDDS', 'FUDS', 'US06'], default='FUDS',
                        help='Choose the source domain')
    parser.add_argument('--model', type=str, choices=['SPP', 'Conv', 'Vit'], default='Conv',
                        help='Choose which model to use')
    parser.add_argument('--save_model', action='store_true', help='Save model after training')
    parser.add_argument('--save_res', action='store_true', help='Save results to CSV file')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--counts', type=int, default=32, help=' ')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train_data, train_label, val_data, val_label = dataset_get.Generate_Dataset_from_Dir("./Dataset", "UDDS")
    train_data, train_label, val_data, val_label = torch.from_numpy(train_data), torch.from_numpy(
        train_label), torch.from_numpy(val_data), torch.from_numpy(val_label)
    train_data_F, train_label_F, val_data_F, val_label_F = dataset_get.Generate_Dataset_from_Dir("./Dataset", "FUDS")
    train_data_F, train_label_F, val_data_F, val_label_F = torch.from_numpy(train_data_F), torch.from_numpy(
        train_label_F), torch.from_numpy(val_data_F), torch.from_numpy(val_label_F)
    train_data_6, train_label_6, val_data_6, val_label_6 = dataset_get.Generate_Dataset_from_Dir("./Dataset", "US06")
    train_data_6, train_label_6, val_data_6, val_label_6 = torch.from_numpy(train_data_6), torch.from_numpy(
        train_label_6), torch.from_numpy(val_data_6), torch.from_numpy(val_label_6)
    print("Read OK!")

    # Initialize datasets
    train_set = dataset_get.CustomedDataSet(train_x=train_data, train_y=train_label)
    val_set = dataset_get.CustomedDataSet(test_x=val_data, test_y=val_label, train=False, val=True)
    train_set_F = dataset_get.CustomedDataSet(train_x=train_data_F, train_y=train_label_F)
    val_set_F = dataset_get.CustomedDataSet(test_x=val_data_F, test_y=val_label_F, train=False, val=True)
    train_set_6 = dataset_get.CustomedDataSet(train_x=train_data_6, train_y=train_label_6)
    val_set_6 = dataset_get.CustomedDataSet(test_x=val_data_6, test_y=val_label_6, train=False, val=True)
    print("Dataset OK!")

    # Initialize DataLoaders
    train_loader = dataset_get.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = dataset_get.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
    train_loader_F = dataset_get.DataLoader(dataset=train_set_F, batch_size=args.batch_size, shuffle=True)
    val_loader_F = dataset_get.DataLoader(dataset=val_set_F, batch_size=args.batch_size, shuffle=False)
    train_loader_6 = dataset_get.DataLoader(dataset=train_set_6, batch_size=args.batch_size, shuffle=True)
    val_loader_6 = dataset_get.DataLoader(dataset=val_set_6, batch_size=args.batch_size, shuffle=False)
    print("DataLoader OK!")

    # Select model based on argument
    if args.model == 'SPP':
        model = model_SPP.SELF()
        model_SPP.initialize_weights(model)
    elif args.model == 'Conv':
        model = model_Conv.SELF()
        model_Conv.initialize_weights(model)
    elif args.model == 'Vit':
        model = model_Vit.SELF()
        model_Vit._init_vit_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    network = model.to(device)
    network.double()

    loss_ave_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    val_accuracy_list_F = []
    val_accuracy_list_6 = []
    train_classify_corr_list = []
    val_classify_corr_list = []
    val_classify_corr_list_F = []
    val_classify_corr_list_6 = []

    # Select condition based on command line argument
    if args.select_condition == "UDDS":
        train_loader_s = train_loader
    elif args.select_condition == "FUDS":
        train_loader_s = train_loader_F
    elif args.select_condition == "US06":
        train_loader_s = train_loader_6

    for i in range(1, args.n_epochs + 1):
        print(f"Epoch {i}\n-------------------------------")
        network.train()

        loss_ave = train(train_loader=train_loader_s, optimizer=optimizer, network=network)
        loss_ave_list.append(loss_ave)

        network.eval()
        train_accuracy, train_classify_corr = accuracy(epoch_idx=i, test_loader=train_loader_s, network=network,
                                                       set_type="train")
        train_accuracy_list.append(train_accuracy)
        train_classify_corr_list.append(train_classify_corr)

        print("Testing in UDDS")
        val_accuracy, val_classify_corr = accuracy(epoch_idx=i, test_loader=val_loader, network=network, set_type="val")
        val_accuracy_list.append(val_accuracy)
        val_classify_corr_list.append(val_classify_corr)

        print("Testing in FUDS")
        val_accuracy_F, val_classify_corr_F = accuracy(epoch_idx=i, test_loader=val_loader_F, network=network,
                                                       set_type="val")
        val_accuracy_list_F.append(val_accuracy_F)
        val_classify_corr_list_F.append(val_classify_corr_F)

        print("Testing in US06")
        val_accuracy_6, val_classify_corr_6 = accuracy(epoch_idx=i, test_loader=val_loader_6, network=network,
                                                       set_type="val")
        val_accuracy_list_6.append(val_accuracy_6)
        val_classify_corr_list_6.append(val_classify_corr_6)

    # Save model if --save_model is specified
    if args.save_model:
        torch.save(network.state_dict(), "./Model/Default.pt")

    # Plot accuracy and loss
    plt.plot(range(len(train_accuracy_list)), train_accuracy_list, color='b', label='Train_accuracy')
    plt.plot(range(len(val_accuracy_list)), val_accuracy_list, color='g', label='UDDS_Val_accuracy')
    plt.plot(range(len(val_accuracy_list_F)), val_accuracy_list_F, color='r', label='FUDS_Val_accuracy')
    plt.plot(range(len(val_accuracy_list_6)), val_accuracy_list_6, color='brown', label='US06_Val_accuracy')
    plt.legend()
    # plt.show()

    # Save results
    if args.save_res:
        count = args.counts
        All_list = [
            [loss_ave_list[i], train_accuracy_list[i], val_accuracy_list[i], val_accuracy_list_F[i],
             val_accuracy_list_6[i]]
            for i in range(args.n_epochs)]

        df1 = pd.DataFrame(All_list, columns=['loss', 'Train', 'Val[UDDS]', 'Val[FUDS]', 'Val[US06]'])

        # 动态生成保存路径，使用<model>_Square作为文件夹名称
        save_path = f'./Result_csv/{args.model}_Square/{args.select_condition}_Res_{count}.csv'

        # 检查是否保存
        df1.to_csv(save_path, index=False)
        print(f"Successfully Results saved to {save_path}")

        All_classify_list = []
        for i in range(0, args.n_epochs):
            All_classify_list.append(
                [train_classify_corr_list[i][0], train_classify_corr_list[i][1], train_classify_corr_list[i][2],
                 train_classify_corr_list[i][3], train_classify_corr_list[i][4],
                 val_classify_corr_list[i][0], val_classify_corr_list[i][1], val_classify_corr_list[i][2],
                 val_classify_corr_list[i][3], val_classify_corr_list[i][4],
                 val_classify_corr_list_F[i][0], val_classify_corr_list_F[i][1], val_classify_corr_list_F[i][2],
                 val_classify_corr_list_F[i][3], val_classify_corr_list_F[i][4],
                 val_classify_corr_list_6[i][0], val_classify_corr_list_6[i][1], val_classify_corr_list_6[i][2],
                 val_classify_corr_list_6[i][3], val_classify_corr_list_6[i][4]])
        df2 = pd.DataFrame(data=All_classify_list,
                           columns=['Cor[Train]', 'ISC[Train]', 'Noi[Train]', 'Nor[Train]', 'Vis[Train]',
                                    'Cor[UDDS]', 'ISC[UDDS]', 'Noi[UDDS]', 'Nor[UDDS]', 'Vis[UDDS]',
                                    'Cor[FUDS]', 'ISC[FUDS]', 'Noi[FUDS]', 'Nor[FUDS]', 'Vis[FUDS]',
                                    'Cor[US06]', 'ISC[US06]', 'Noi[US06]', 'Nor[US06]', 'Vis[US06]'])

        save_path2 = f'./Result_csv/{args.model}/{args.select_condition}_Res_Classify_{count}.csv'
        df2.to_csv(save_path2, index=False)
        print(f"Successfully Results saved to {save_path2}")

