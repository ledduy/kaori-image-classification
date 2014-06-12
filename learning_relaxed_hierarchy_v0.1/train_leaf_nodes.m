function model = train_leaf_nodes(model, Y, X, config)

if size(Y, 1) < size(Y, 2)
  Y = Y';
end

num_level = model.node_models{end}.level;
num_current_model = length(model.node_models);
node_cnt = num_current_model;

options = [' -q -t 4 -s 0 -c ' num2str(config.C_one_vs_all)];

num_proc = config.num_proc;
if num_proc > 1
  matlabpool(num_proc);
end

for k =  num_current_model : -1 : 1
  if model.node_models{k}.level ~= num_level
    break;
  end

  disp(['processing node ' num2str(k) ' ...']);
  mu = model.node_models{k}.mu;
    
  % left node
  active_classes_binary = (mu <= 0) & (mu > -2);
  active_classes = find(active_classes_binary == 1);
  if length(active_classes) >= 2
    node_cnt = node_cnt + 1;

    y_binary = mu(Y);
    if size(y_binary, 1) < size(y_binary, 2)
      y_binary = y_binary';
    end
    
    train_indx = [];
    
    for i = 1 : length(active_classes)
      train_indx = [train_indx; find(Y == active_classes(i))];
    end
    y_space = unique(Y(train_indx));
    Y_new = Y(train_indx);
    for i = 1 : length(y_space)
      Y_new(Y_new==y_space(i)) = i;
    end
    
    leaf_model = train_1vsAll(double(Y_new), double(X(train_indx, train_indx)), options, true, config.num_proc);
    
    for i = 1 : length(leaf_model)
      leaf_model{i}.global_SV_indice = train_indx(leaf_model{i}.SVs);
    end
    
    node_cnt = node_cnt + 1;
    model.node_models{node_cnt}.binary_model = leaf_model;
    model.node_models{node_cnt}.level = num_level + 1;
    model.node_models{node_cnt}.right_indx = [];
    model.node_models{node_cnt}.left_indx = [];

    model.node_models{node_cnt}.mu = -2 * ones(length(mu), 1);
    model.node_models{node_cnt}.mu(active_classes) = 1;
    
    model.node_models{k}.left_indx = node_cnt;
  end
  
  % right
  active_classes = find(mu >= 0);
  if length(active_classes) >= 2
    node_cnt = node_cnt + 1;
    
    y_binary = mu(Y);
    if size(y_binary, 1) < size(y_binary, 2)
      y_binary = y_binary';
    end
    
    train_indx = [];
    
    for i = 1 : length(active_classes)
      train_indx = [train_indx; find(Y == active_classes(i))];
    end
    y_space = unique(Y(train_indx));
    Y_new = Y(train_indx);
    for i = 1 : length(y_space)
      Y_new(Y_new==y_space(i)) = i;
    end
    
    leaf_model = train_1vsAll(double(Y_new), double(X(train_indx, train_indx)), options, true, config.num_proc);
    
    
    for i = 1 : length(leaf_model)
      leaf_model{i}.global_SV_indice = train_indx(leaf_model{i}.SVs);
    end
    
    node_cnt = node_cnt + 1;
    model.node_models{node_cnt}.binary_model = leaf_model;
    model.node_models{node_cnt}.level = num_level + 1;
    model.node_models{node_cnt}.right_indx = [];
    model.node_models{node_cnt}.left_indx = [];

    model.node_models{node_cnt}.mu = -2 * ones(length(mu), 1);
    model.node_models{node_cnt}.mu(active_classes) = 1;
    
    model.node_models{k}.right_indx = node_cnt;
  end

end


if num_proc > 1
  matlabpool close;
end