function [Y_pred dec_values] = test_1vsAll(X, model, varargin)

b_use_global_indice = false;
if size(varargin,2) >= 1
  b_use_global_indice = varargin{1};
end

num_classes = length(model);
num_test_inst = size(X, 2);

resp = zeros(num_test_inst, num_classes);

for c = 1 : num_classes
  resp(:, c) = eval_binary_model_K(X, model{c}, b_use_global_indice);
end

[dec_values Y_pred] = max(resp, [], 2);



