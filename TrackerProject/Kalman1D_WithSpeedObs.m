function Kalman1D_WithSpeedObs
    dt = 1;
    N = 50;
    true_velocity = 1.0;
    Q = 0.01 * eye(2);         % 过程噪声
    R = diag([1.0, 0.5]);      % 测量噪声：位置 1.0，速度 0.5

    F = [1 dt; 0 1];           % 状态转移矩阵
    H = eye(2);                % 观测矩阵：我们同时测量 p 和 v

    x_true = [0; true_velocity];
    x_est = [0; 0];
    P = eye(2);

    x_store = zeros(2, N);
    z_store = zeros(2, N);

    for k = 1:N
        % 模拟真实状态 & 测量
        w = mvnrnd([0; 0], Q)';         % 过程噪声
        v = mvnrnd([0; 0], R)';         % 测量噪声
        x_true = F * x_true + w;
        z = H * x_true + v;

        % Kalman 预测
        x_pred = F * x_est;
        P_pred = F * P * F' + Q;

        % Kalman 更新
        K = P_pred * H' / (H * P_pred * H' + R);
        x_est = x_pred + K * (z - H * x_pred);
        P = (eye(2) - K * H) * P_pred;

        % 存储
        x_store(:, k) = x_est;
        z_store(:, k) = z;
    end

    % 画图
    t = 1:N;
    figure;
    subplot(2, 1, 1);
    plot(t, x_store(1, :), 'b-', 'LineWidth', 2); hold on;
    plot(t, z_store(1, :), 'r.');
    legend('KF 估计位置', '测量位置');
    title('位置估计');

    subplot(2, 1, 2);
    plot(t, x_store(2, :), 'g-', 'LineWidth', 2); hold on;
    plot(t, z_store(2, :), 'm.');
    legend('KF 估计速度', '测量速度');
    title('速度估计');
end