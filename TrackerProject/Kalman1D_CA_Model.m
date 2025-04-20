function Kalman1D_CA_Model
clc;clear all;close all;
dt = 1;
N = 50;
true_a = 0.2;             % 真实加速度
Q = 0.00000001 * eye(3);        % 过程噪声（稍微调大一点以适应模型不匹配）
R = diag([1.0, 0.5]);     % 测量噪声（测位置 + 速度）

% 状态转移矩阵 F（CA 模型）
F = [1 dt 0.5*dt^2;
    0  1     dt;
    0  0     1];

% 观测矩阵 H：测位置 + 速度
H = [1 0 0;
    0 1 0];

% 初始状态
x_true = [0; 0; true_a];
x_est  = [0; 0; 0];        % 初始估计
P = eye(3);                % 协方差

x_store = zeros(3, N);
z_store = zeros(2, N);

for k = 1:N
    x_true_store(:, k) = x_true;  % 增加这一行，存储真实轨迹
    % 模拟真实运动 + 噪声测量
    w = mvnrnd([0; 0; 0], Q)';
    v = mvnrnd([0; 0], R)';
    x_true = F * x_true + w;
    z = H * x_true + v;

    % KF 预测
    x_pred = F * x_est;
    P_pred = F * P * F' + Q;

    % KF 更新
    K = P_pred * H' / (H * P_pred * H' + R);
    x_est = x_pred + K * (z - H * x_pred);
    P = (eye(3) - K * H) * P_pred;

    % 存储
    x_store(:, k) = x_est;
    z_store(:, k) = z;
end

% 绘图
t = 1:N;
figure;
subplot(3, 1, 1);
plot(t, x_store(1, :), 'b-', 'LineWidth', 2); hold on;      % 滤波估计
plot(t, z_store(1, :), 'r.', 'MarkerSize', 10);hold on;     % 带噪测量
plot(t, x_true_store(1, :), 'k--', 'LineWidth', 2);hold on;  % 真实轨迹
legend('KF 估计位置', '测量位置', '真实位置');
title('位置');

subplot(3, 1, 2);
plot(t, x_store(2, :), 'g-', 'LineWidth', 2); hold on;
plot(t, z_store(2, :), 'm.');hold on;
plot(t, x_true_store(2, :), 'k--');hold on;  % 真实速度
legend('估计速度', '测量速度','真实速度'); title('速度');

subplot(3, 1, 3);
plot(t, x_store(3, :), 'k-', 'LineWidth', 2); hold on;
yline(true_a, '--r');
legend('估计加速度', '真实加速度'); title('加速度');
end