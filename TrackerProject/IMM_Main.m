function IMM_Main
clc;clear all;close all;
% ====== 通用设置 ======
N = 50;               % 步数
dt = 1;
x_true = [0; 1; 0.2]; % 初始真实状态
z_dim = 2;
x_dim = 3;

% 状态转移矩阵
F_CV = [1 dt 0; 0 1 0; 0 0 1];                     % CV: 匀速
Q_CV = diag([0.05, 0.05, 1e-6]);                  % 加速度保持极小变化

F_CA = [1 dt 0.5*dt^2; 0 1 dt; 0 0 1];             % CA: 匀加速
Q_CA = diag([0.05, 0.05, 0.01]);                  % 正常加速度过程噪声

H = [1 0 0; 0 1 0];           % 测量位置 & 速度
R = 10 * diag([1.0, 0.5]);         % 测量噪声

% 模型数 M = 2
M = 2;
x_est = repmat([0; 0; 0], 1, M);     % 每列为一个模型状态估计
P_est = repmat(eye(x_dim), 1, 1, M); % P 是 3维数组
mu = [0.5; 0.5];                     % 初始模型概率
Pi = [0.9 0.1; 0.1 0.9];             % 模型转移矩阵

% 数据存储
x_true_store = zeros(x_dim, N);
x_fused_store = zeros(x_dim, N);
mu_store = zeros(M, N);

% ====== 主循环 ======
for k = 1:N
    % === 模拟真实运动（用 CA 模型） ===
    w = mvnrnd([0; 0; 0], 0.01 * eye(3))';
    x_true = F_CA * x_true + w;
    z = H * x_true + mvnrnd([0; 0], R)';

    % === 1. 模型混合（mixing） ===
    x_mix = zeros(x_dim, M);
    P_mix = zeros(x_dim, x_dim, M);
    c_j = Pi' * mu;  % 混合权重归一化因子
    for j = 1:M
        for i = 1:M
            mu_ij = Pi(i,j) * mu(i) / c_j(j);  % 条件概率
            x_mix(:, j) = x_mix(:, j) + mu_ij * x_est(:, i);
        end
    end
    % 协方差混合
    for j = 1:M
        for i = 1:M
            mu_ij = Pi(i,j) * mu(i) / c_j(j);
            dx = x_est(:,i) - x_mix(:,j);
            P_mix(:,:,j) = P_mix(:,:,j) + ...
                mu_ij * (P_est(:,:,i) + dx * dx');
        end
    end

    % === 2. 对每个模型单独滤波 ===
    modelF = {F_CV, F_CA};
    modelQ = {Q_CV, Q_CA};
    likelihoods = zeros(M,1);
    enablePDAF = 0;
    for j = 1:M
        if enablePDAF == 1
            [x_est(:,j), P_est(:,:,j), beta_all] = PDAF_Update(x_pred, P_pred, Z_all, H, R, P_FA);
            likelihood(j) = sum(beta_all);  % 可以直接使用权重和作为观测解释力
        else

            [x_est(:,j), P_est(:,:,j), likelihoods(j)] = ...
                KalmanModel(x_mix(:,j), P_mix(:,:,j), z, ...
                modelF{j}, H, modelQ{j}, R);
        end
    end
    % === 3. 更新模型概率（贝叶斯融合） ===
    mu_temp = (likelihoods .* (Pi' * mu))';
    mu = mu_temp' / sum(mu_temp);

    % === 4. 融合最终输出 ===
    x_fused = x_est * mu;
    P_fused = zeros(x_dim);
    for j = 1:M
        dx = x_est(:,j) - x_fused;
        P_fused = P_fused + mu(j) * (P_est(:,:,j) + dx * dx');
    end

    % === 数据存储 ===
    x_true_store(:,k) = x_true;
    x_fused_store(:,k) = x_fused;
    mu_store(:,k) = mu;
    P_fused_store(:,:,k) = P_fused;
end

% ====== 绘图展示 ======
t = 1:N;
figure;
subplot(3,1,1); hold on;
plot(t, x_true_store(1,:), 'k--');
plot(t, x_fused_store(1,:), 'b-'); title('位置');

subplot(3,1,2); hold on;
plot(t, x_true_store(2,:), 'k--');
plot(t, x_fused_store(2,:), 'g-'); title('速度');

subplot(3,1,3); hold on;
plot(t, mu_store(1,:), 'r-', 'LineWidth', 1.5);
plot(t, mu_store(2,:), 'b--', 'LineWidth', 1.5);
legend('CV模型概率','CA模型概率'); title('模型概率');

t = 1:N;

% 提取位置估计误差标准差
pos_std = zeros(1, N);
for k = 1:N
    pos_std(k) = sqrt(P_fused_store(1,1,k));  % 位置维度的方差
end

% 可视化位置 + 置信区间
figure;
plot(t, x_true_store(1,:), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_fused_store(1,:), 'b-', 'LineWidth', 2);
plot(t, x_fused_store(1,:) + pos_std, 'r--');
plot(t, x_fused_store(1,:) - pos_std, 'r--');
legend('真实位置', '估计位置', '+1σ区间', '-1σ区间');
xlabel('时间'); ylabel('位置'); title('IMM 位置估计及置信带');
grid on;

% 使用 trace(P) 评估整体不确定性
P_trace = zeros(1, N);
for k = 1:N
    P_trace(k) = trace(P_fused_store(:,:,k));
end

figure;
plot(t, P_trace, 'm-', 'LineWidth', 2);
xlabel('时间'); ylabel('Trace(P)');
title('IMM 融合协方差矩阵的不确定性 (trace)');
grid on;
end