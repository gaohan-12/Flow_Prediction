# This module is designed for the optical flow prediction,
# which contains the model function and train strategy.

import torch
import os
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
import torch.utils.data as data
from flowlib import read_flow, write_flow

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


class BiConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):

        super(BiConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4*self.hidden_dim,
                              kernel_size=self.kernel_size,
                              bias=self.bias,
                              padding=self.padding)

        self.conv_concat = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              bias=self.bias,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):

        c_cur, h_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1).cuda()     # 将输入与隐藏层结合
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)

        f = torch.sigmoid(cc_f)   # 遗忘门
        i = torch.sigmoid(cc_i)   # 输入门
        g = torch.tanh(cc_g)      # 当前细胞信息
        o = torch.sigmoid(cc_o)   # 输出门
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next


class BiConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True, return_all_layers=False):

        super(BiConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(BiConvLSTMCell(input_size=(self.height, self.width),
                                            input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias).cuda())
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):

        hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        # print(seq_len)

        for layer_idx in range(self.num_layers):
            backward_states = []
            forward_states = []
            output_inner = []

            hb, cb = hidden_state[layer_idx]
            # print('hb,cb',hb.shape,cb.shape)
            for t in range(seq_len):

                hb, cb = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, seq_len - t - 1, :, :, :], cur_state=[hb, cb])

                backward_states.append(hb)

            hf, cf = hidden_state[layer_idx]
            for t in range(seq_len):
                hf, cf = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[hf, cf])
                # print('hf:',hf.shape)
                forward_states.append(hf)

            for t in range(seq_len):

                h = self.cell_list[layer_idx].conv_concat(torch.cat((forward_states[t], backward_states[seq_len - t - 1]), dim=1))

                # print('h',h.shape)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)

        if not self.return_all_layers:
            return layer_output_list[-1]          # 取最后一个元素

        return layer_output_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
                init_states.append((Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)).cuda(),
                                    Variable(torch.zeros(batch_size, self.hidden_dim[i], self.height, self.width)).cuda()))
        return init_states

    # 判断是否为元组或元组列表
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class MyDataset(data.Dataset):

    def __init__(self, path_dir, transform=None):

        self.path_dir = path_dir
        self.transform = transform
        self.flow = os.listdir(self.path_dir)
        self.flow = sorted(self.flow, key=lambda x: int(x[:-4]))

    def __getitem__(self, index):

        if index < len(self.flow) - 1:
            flow_index = self.flow[index]   # 根据索引获取光流文件名称
            label_index = self.flow[index+1]       # 根据索引获取标签名称
            flow_path = os.path.join(self.path_dir, flow_index)  # 获取光流的路径或目录
            label_path = os.path.join(self.path_dir, label_index)    # 获取标签的路径或目录
            flow = read_flow(flow_path)   # 读取光流
            label = read_flow(label_path)   # 读取标签

        else:
            flow_index = self.flow[index]
            flow_path = os.path.join(self.path_dir, flow_index)
            flow = read_flow(flow_path,)
            label = flow

        # if self.transform is not None:
        #     flow = self.transform(flow)
        #     label = self.transform(label)

        return flow, label

    def __len__(self):
        return len(self.flow)


data_transform = transforms.Compose([transforms.ToTensor()])
data_root = os.path.abspath(os.path.join(os.getcwd())) + "/dataset/"

flow_test = read_flow(data_root + "/train/1/0.flo")
print(flow_test.shape)
(height, width, channel) = flow_test.shape


model = BiConvLSTM(input_size=(height, width), input_dim=channel, hidden_dim=channel, kernel_size=(3, 3), num_layers=1)
# model.to(device)

epoch_size = 16
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def Var(x):
    return Variable(x.cuda())


def train(self):

    for epoch in range(epoch_size):

        for i in range(1, 13):

            train_path_i = data_root + "/train/" + str(i) + "/"
            train_dataset_i = MyDataset(train_path_i, transform=data_transform)
            train_dataloader_i = DataLoader(train_dataset_i, batch_size=1, shuffle=False)

            for step, data in enumerate(train_dataloader_i):

                optimizer.zero_grad()

                input_tensor = Var(data[0])
                # print(input_tensor.shape)
                ref_tensor = Var(data[1])
                # print(ref_tensor.shape)

                input_tensor = input_tensor.reshape(1, 1, channel, height, width)
                ref_tensor = ref_tensor.reshape(1, 1, channel, height, width)

                output = model(input_tensor).cuda()
                # print(output.shape)

                loss = criterion(output, ref_tensor)
                loss.backward()
                optimizer.step()

                if step % 20 == 0:
                    print(datetime.now().strftime('%c'), 'train:', epoch, step, 'loss:', loss)

                if epoch > epoch_size - 2:
                    if os.path.exists("./train_results_flow/" + str(epoch) + "/" + str(i) + "/") is False:
                        os.makedirs('train_results_flow/' + str(epoch) + '/' + str(i) + '/')

                    write_flow(output.reshape(height, width, channel), "./train_results_flow/" + str(epoch) + '/' + str(i) + '/' + str(step) + '.flo')

    torch.save(model, 'model.pth')


def val(self):

    for j in range(13, 25):

        val_path_j = data_root + "/val/" + str(j) + "/"
        val_dataset_j = MyDataset(val_path_j, transform=data_transform)
        val_dataloader_j = DataLoader(val_dataset_j, batch_size=1, shuffle=False)

        for step, data in enumerate(val_dataloader_j):

            input_tensor = Var(data[0])
            ref_tensor = Var(data[1])

            input_tensor = input_tensor.reshape(1, 1, channel, height, width)
            ref_tensor = ref_tensor.reshape(1, 1, channel, height, width)

            output = model(input_tensor).cuda()
            loss = criterion(output, ref_tensor)

            if step % 20 == 0:
                print(datetime.now().strftime('%c'), 'val:', step, 'loss:', loss)

            if os.path.exists("./val_results_flow/" + str(j) + "/") is False:
                os.makedirs('val_results_flow/' + str(j) + '/')

            write_flow(output.reshape(height, width, channel), "./val_results_flow/" + str(j) + '/' + str(step) + '.flo')


if __name__ == "__main__":
    train(model)
    val(model)
