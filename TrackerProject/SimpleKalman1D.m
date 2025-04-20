function SimpleKalman1D
    % 参数设定
    dt = 1;                % 时间步长
    N = 50;                % 仿真总步数
    true_velocity = 1.0;   % 真实速度
    Q = 0.01 * eye(2);     % 过程噪声协方差（系统的不确定性）
    R = 1.0;               % 测量噪声协方差（测量误差）

    % 模型矩阵
    F = [1 dt; 0 1];       % 状态转移矩阵
    H = [1 0];             % 观测矩阵（我们只测量位置）

    % 初始状态
    x_true = [0; true_velocity];   % 真实目标状态（位置+速度）
    x_est = [0; 0];                % 初始估计（你一开始不知道速度）
    P = eye(2);                    % 初始协方差矩阵（估计误差大）

    % 数据存储
    x_store = zeros(2, N);         % 存状态估计
    z_store = zeros(1, N);         % 存测量

    for k = 1:N
        % ------ 模拟真实运动和观测 ------
        w = mvnrnd([0; 0], Q)';     % 模拟过程噪声 w_k ~ N(0, Q)
        v = sqrt(R) * randn;        % 模拟测量噪声 v_k ~ N(0, R)

        x_true = F * x_true + w;    % 真实目标按照状态转移运动（+扰动）
        z = H * x_true + v;         % 测量只有位置，带有误差

        % ------ KF 预测步骤 ------
        x_pred = F * x_est;         % 状态预测
        P_pred = F * P * F' + Q;    % 协方差预测

        % ------ KF 更新步骤 ------
        K = P_pred * H' / (H * P_pred * H' + R);    % 卡尔曼增益
        x_est = x_pred + K * (z - H * x_pred);      % 利用观测更新状态估计
        P = (eye(2) - K * H) * P_pred;             % 更新协方差矩阵

        % ------ 保存估计数据 ------
        x_store(:, k) = x_est;      % 保存每一帧的估计
        z_store(k) = z;             % 保存每一帧的观测值
    end

    % ------ 绘图展示 ------
    t = 1:N;
    figure;
    plot(t, x_store(1, :), 'b-', 'LineWidth', 2); hold on;
    plot(t, z_store, 'r.', 'MarkerSize', 10);
    legend('KF Estimated Position', 'Measurements');
    xlabel('Time step'); ylabel('Position');
    title('1D Kalman Filter Tracking');
    grid on;
end