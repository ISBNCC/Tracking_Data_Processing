function [x_upd, P_upd, likelihood] = KalmanModel(x0, P0, z, F, H, Q, R)
    % Prediction
    x_pred = F * x0;
    P_pred = F * P0 * F' + Q;

    % Innovation (residual)
    y = z - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;

    % Update
    x_upd = x_pred + K * y;
    P_upd = (eye(size(K,1)) - K * H) * P_pred;

    % Likelihood calculation (for model probability update)
    likelihood = exp(-0.5 * y' / S * y) / sqrt((2*pi)^length(y) * det(S));
end