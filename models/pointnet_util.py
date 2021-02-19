import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):    #归一化
    l = pc.shape[0] #点云长度
    centroid = np.mean(pc, axis=0) #对各列求均值
    pc = pc - centroid #减均值
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) #求最大
    pc = pc / m #求归一化
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points. #计算两组点云每两点之间的欧氏距离
    其中src和dst的shape例如tensor([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]],
                           [[x4,y4,z4],[x5,y5,z5],[x6,y6,z6]])
    B = 2，N =3，C=3
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M] #B为batchsize，N为第一组点s数量，M为第二组点数量，C为通道数3
    """
    B, N, _ = src.shape #赋值给B，N
    _, M, _ = dst.shape #赋值给M
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))#对应-2*src^T*dst= xn * xm + yn * ym + zn * zm，其中permute是为了求转置，matmul是乘法
    dist += torch.sum(src ** 2, -1).view(B, N, 1) #sum(src**2,dim=-1)，view维度从xyz变成1维才能相乘求和
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) #sum(dst**2,dim=-1)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C] #N为每个点云的点数目，原始各个点云N假设为2048
        idx: sample index data, [B, S]#输入参数S为[1,333,1000,2000],新的N=4，即从B个样本中取每个样本的取第1个点，第二个点云中取第333个点，第三个点云中取第1000个点，第四个点云中取第2000个点，新的点云集为B*4*3
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint): #采样点之间距离足够远
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) #初始化采样点矩阵B*npoint零矩阵，npoint为采样点数
    distance = torch.ones(B, N).to(device) * 1e10 #初始化距离，B*npoints矩阵每个值都是1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) #随机初始化最远点，随机数范围是从0-N，一共是B个，维度是1*B，保证每个B都有一个最远点
    batch_indices = torch.arange(B, dtype=torch.long).to(device) #0~(B-1)的数组
    for i in range(npoint):#寻找并选取空间中每个点距离多个采样点的最短距离，并存储在dist
        centroids[:, i] = farthest #设采样点为farthers点,[:,i]为取所有行的第i个数据
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) #取中心点也是farthest点
        dist = torch.sum((xyz - centroid) ** 2, -1) #求所有点到farthest点的欧式距离和
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]#返回最大距离的点
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz): #寻找球半径里面的点，从S个球内采样nsample个点
    """
    Input:
        radius: local region radius球半径
        nsample: max sample number in local region每个球所要采样的点数
        xyz: all points, [B, N, 3]，全部点
        new_xyz: query points, [B, S, 3]，S个球形领域中心点，由farthestpoint组成
    Return:
        group_idx: grouped points index, [B, S, nsample]，输出球形领域采样点索引
    """
    device = xyz.device
    B, N, C = xyz.shape #原始点云的BNC
    _, S, _ = new_xyz.shape #由index_points得出的S，例如一共有S个球
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) #获取点云的各个点的序列位置
    sqrdists = square_distance(new_xyz, xyz)#计算中心点与所有点的欧式距离
    group_idx[sqrdists > radius ** 2] = N #大于欧氏距离平方的点序列标签设置为N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#升序排列，N是最大值，剩下的点为设定点数nsample的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])#用第一个点代替nsample个点中被赋值为N的点
    mask = group_idx == N #把N点也替换成第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):#每个点云被分割成group局部区域也就是每个球，使用Pointnet计算每个group全局特征
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C] fathest_point_sample函数索引到采样点
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx) #通过index_points函数将最远点从原始点云中挑选出来作为新的xyz
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)#将原始点云分割为每个球体，每个球体有nsample个采样点
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)#每个球体区域的点减去中心点
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points): #将所有点作为一个group，和上面相同
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i] #对获取到的点做MLP操作
            new_points =  F.relu(bn(conv(new_points))) #归一化操作

        new_points = torch.max(new_points, 2)[0] #最大池化得到全局特征
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module): #MSG层，相比于普通的radius，这里是radius_list，例如[0.1,0.2,0.4],针对不同的半径做ball query，最后将不同半径下的点云特征保存在list里面，再拼接在一起
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list): #针对不同半径，做不同的ball query
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points) #拼接点云

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else: #线性插值，上采样
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm #距离越远的点权重越小
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) #每个点的权重再归一化

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

