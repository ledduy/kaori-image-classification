function node_model = train_node_model(Y, X, confusion_matrix, config, global_indx)

% re-index the label vector
y_space = unique(Y);
num_classes = length(y_space);
Y_new = Y;
for i = 1 : num_classes
    Y_new(Y==y_space(i)) = i;
end

best_bin_model = [];
best_mu = [];
best_score = +inf;
best_node_train_indx = [];

% -----------------------------------------------------
%                          get multiple initializations
% -----------------------------------------------------
switch(lower(config.init_method)) 
 case 'top_down'
  mu_init = get_top_down_initialization(X, Y, config.alpha);
 case 'bottom_up'
  mu_init = get_bottom_up_initialization(confusion_matrix, config.num_samples);
 otherwise
  mu_init = get_bottom_up_initialization(confusion_matrix, config.num_samples);
end  



% -----------------------------------------------------
% train different models with different initializations
% -----------------------------------------------------
for i = 1 : length(mu_init)
  % learn the model using the initial coloring
  [mu bin_model] = train_node_model_with_initialization(Y_new, X, mu_init{i}, config);

  % model selection
  if strcmp(config.kernel_type, 'linear')
    score = compute_objective(Y_new, X, mu, bin_model, config);
  else
    score = compute_average_num_sv(mu, bin_model);
  end
  
  % get training indices
  y_color = mu(Y_new);
  node_train_indx = find(y_color ~= 0);
  
  % keep track of the best model
  if score < best_score
    best_score = score;
    best_mu = mu;
    best_bin_model = bin_model;
    best_node_train_indx = node_train_indx;
  end
  
end

% get the global indices for the support vectors
tmp_indice = global_indx(best_node_train_indx);
best_bin_model.global_SV_indice = tmp_indice(best_bin_model.SVs);

node_model.binary_model = best_bin_model;
node_model.mu = best_mu;

