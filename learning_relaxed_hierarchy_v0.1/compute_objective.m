function obj = compute_objective(Y, K, mu, model, config)
% assume the range of Y is from 1 to K (num_classes)

% assume the range of Y is from 1 to K
non_zero_indx = [];

y_binary = mu(Y);
non_zero_indx = find(y_binary ~= 0);

resp = eval_binary_model_K(K(:, non_zero_indx), model);
n = length(non_zero_indx);
xi = max([zeros(n, 1) ones(n, 1)-y_binary(non_zero_indx).*resp], [], 2);


% 1/2*|w|^2 + C * sum(|mu_{y_i}|\xi_i) - A * sum(|mu_{y_i}|) 
C = config.C;
A = config.rho * C;
obj = 1/2*model.w_norm_square ...
      + C * sum(xi) ...
      - A * length(non_zero_indx);
