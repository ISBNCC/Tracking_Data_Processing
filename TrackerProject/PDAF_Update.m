function [x_upd, P_upd, beta_all] = PDAF_Update(x_pred, P_pred, Z, H, R, P_FA)
% 输入：
%   x_pred: 预测状态 [n x 1]
%   P_pred: 预测协方差 [n x n]
%   Z     : 所有观测 [m x z_dim]，每行为一个观测
%   H     : 观测矩阵 [z_dim x n]
%   R     : 观测噪声协方差 [z_dim x z_dim]
%   P_FA  : 假警概率，常设为 0.01 ~ 0.1
% 输出：
%   x_upd: 更新后的状态估计
%   P_upd: 更新后的协方差估计
%   beta_all: 各观测的关联概率

[n, ~] = size(x_pred);
[m, z_dim] = size(Z);

% 观测预测
S = H * P_pred * H' + R;
K = P_pred * H' / S;

% 计算每个观测的残差和似然
nu_all = zeros(z_dim, m);
L_all = zeros(1, m);
for i = 1:m
    nu = Z(i,:)' - H * x_pred;
    nu_all(:,i) = nu;
    L_all(i) = exp(-0.5 * nu' / S * nu) / sqrt((2*pi)^z_dim * det(S));
end

% 加入未观测情况（假警权重）
L_total = sum(L_all) + P_FA;
beta_all = L_all / L_total;  % 各观测为目标的概率

% 残差加权平均
nu_bar = sum(nu_all .* beta_all, 2);

% 更新状态
x_upd = x_pred + K * nu_bar;

% 计算协方差修正项（残差离散性）
S_bar = zeros(z_dim);
for i = 1:m
    diff = nu_all(:,i) - nu_bar;
    S_bar = S_bar + beta_all(i) * (diff * diff');
end

P_upd = P_pred - K * S * K' + K * S_bar * K';
end
