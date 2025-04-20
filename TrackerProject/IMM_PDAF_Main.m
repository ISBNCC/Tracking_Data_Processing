function IMM_PDAF_Main
clc;clear all; close all;
% ====== 初始化参数 ======
N = 50;                % 总帧数
dt = 1;                % 时间间隔
x_true = [0; 1; 0.2];  % 真实状态（位置、速度、加速度）
x_dim = 3; z_dim = 2;  % 状态和观测维度
M = 2;                 % 模型数
P_FA = 0.05;           % 假警率（未观测先验）

% CV模型
F_CV = [1 dt 0; 0 1 0; 0 0 1];
Q_CV = diag([0.05, 0.05, 1e-6]);

% CA模型
F_CA = [1 dt 0.5*dt^2; 0 1 dt; 0 0 1];
Q_CA = diag([0.05, 0.05, 0.01]);

H = [1 0 0; 0 1 0];
R = diag([1.0, 0.5]);
Pi = [0.9 0.1; 0.1 0.9];

% 初始值
x_est = repmat([0; 0; 0], 1, M);
P_est = repmat(eye(x_dim), 1, 1, M);
mu = [0.5; 0.5];

% 存储数据
x_true_store = zeros(x_dim, N);
x_fused_store = zeros(x_dim, N);
mu_store = zeros(M, N);
z_true_store = zeros(z_dim, N);
z_fake1_store = zeros(z_dim, N);
z_fake2_store = zeros(z_dim, N);
beta_store = zeros(3, N);  % 存储每帧3个观测的beta权重

% ====== 主循环 ======
for k = 1:N
    % ===== 模拟真实状态 + 多个观测点 =====
%     w = mvnrnd([0;0;0], 0.01*eye(3))';
%     x_true = F_CA * x_true + w;

    if k < 15
        x_true = F_CV * x_true + mvnrnd([0;0;0], 0.01*eye(3))';
    elseif k < 30
        x_true = F_CA * x_true + mvnrnd([0;0;0], 0.01*eye(3))';
    elseif k < 40
        x_true = F_CV * x_true + mvnrnd([0;0;0], 0.01*eye(3))';
    else
        F_CA_neg = [1 dt 0.5*dt^2; 0 1 dt; 0 0 1];  % a为负（刹车）
        x_true = F_CA_neg * x_true + mvnrnd([0;0;0], 0.01*eye(3))';
    end

    % 模拟多个观测点（1个真实 + 2个噪声）
    z_true = H * x_true + mvnrnd([0;0], R)';
    z_fake1 = z_true + [4; -2] + randn(2,1);
    z_fake2 = z_true + [-3; 1] + randn(2,1);
    Z_all = [z_true'; z_fake1'; z_fake2'];

    % 存储观测点
    z_true_store(:,k) = z_true;
    z_fake1_store(:,k) = z_fake1;
    z_fake2_store(:,k) = z_fake2;

    % ===== 模型混合（状态与协方差） =====
    x_mix = zeros(x_dim, M);
    P_mix = zeros(x_dim, x_dim, M);
    c_j = Pi' * mu;
    for j = 1:M
        for i = 1:M
            mu_ij = Pi(i,j) * mu(i) / c_j(j);
            x_mix(:,j) = x_mix(:,j) + mu_ij * x_est(:,i);
        end
    end
    for j = 1:M
        for i = 1:M
            mu_ij = Pi(i,j) * mu(i) / c_j(j);
            dx = x_est(:,i) - x_mix(:,j);
            P_mix(:,:,j) = P_mix(:,:,j) + mu_ij * (P_est(:,:,i) + dx * dx');
        end
    end

    % ===== 每个模型执行 PDAF 更新 =====
    modelF = {F_CV, F_CA};
    modelQ = {Q_CV, Q_CA};
    likelihoods = zeros(M,1);
    for j = 1:M
        % 预测
        x_pred = modelF{j} * x_mix(:,j);
        P_pred = modelF{j} * P_mix(:,:,j) * modelF{j}' + modelQ{j};

        % PDAF 更新
        [x_est(:,j), P_est(:,:,j), beta_all] = PDAF_Update(x_pred, P_pred, Z_all, H, R, P_FA);

        % 当前模型的观测解释力（likelihood）
        likelihoods(j) = sum(beta_all);

        if j == 1
            beta_store(:,k) = beta_all(:);  % 存储CV模型的beta（或改为其他）
        end
    end

    % ===== 模型概率更新（归一化） =====
    mu_temp = (likelihoods .* (Pi' * mu))';
    mu = mu_temp' / sum(mu_temp);

    % ===== 状态融合输出 =====
    x_fused = zeros(x_dim,1);
    for j = 1:M
        x_fused = x_fused + mu(j) * x_est(:,j);
    end

    % ===== 数据记录 =====
    x_true_store(:,k) = x_true;
    x_fused_store(:,k) = x_fused;
    mu_store(:,k) = mu;
end

% ===== 绘图 =====
t = 1:N;
figure;
subplot(4,1,1); hold on;
plot(t, x_true_store(1,:), 'k--');
plot(t, x_fused_store(1,:), 'b-', 'LineWidth', 1.5);
plot(t, z_true_store(1,:), 'ro');
plot(t, z_fake1_store(1,:), 'gx');
plot(t, z_fake2_store(1,:), 'mx');
title('位置'); legend('真实', '估计', '观测真', '干扰1', '干扰2');

subplot(4,1,2); plot(t, x_true_store(2,:), 'k--', t, x_fused_store(2,:), 'g-'); title('速度'); legend('真实', '估计');
subplot(4,1,3); plot(t, mu_store(1,:), 'r-', t, mu_store(2,:), 'b--'); title('模型概率'); legend('CV', 'CA');
subplot(4,1,4); hold on;
plot(t, beta_store(1,:), 'ro-', 'LineWidth', 1);
plot(t, beta_store(2,:), 'gx-');
plot(t, beta_store(3,:), 'mx-');
title('PDAF 观测关联概率（CV模型）');
legend('真观测', '干扰1', '干扰2');
xlabel('时间');
end
