function confusion_matrix = estimate_confusion_matrix(Y, X, config)
% assume the range of Y is from 1 to K (the number of classes)

num_classes = length(unique(Y));
num_fold = 5;

%% estimate the confusion matrix using 5-fold cross validation
rand('seed', sum(100*clock));

% generate random permutation
% and train and test splits
train_indices = cell(num_fold, 1);
test_indices = cell(num_fold, 1);

for k = 1 : num_classes
  class_k_indices = find(Y == k);
  num_instance_per_class = length(class_k_indices);
  rand_indices = randperm(num_instance_per_class);
  num_instace_fold = floor(num_instance_per_class / num_fold);
  
  for f = 1 : num_fold
    test_indices_tmp = [1+(f-1)*num_instace_fold : min(num_instance_per_class, f*num_instace_fold)]';
    train_indices_tmp = setdiff([1:num_instance_per_class], test_indices_tmp);
    test_indices_tmp = rand_indices(test_indices_tmp);
    train_indices_tmp = rand_indices(train_indices_tmp);
    
    test_indices{f} = [test_indices{f}; class_k_indices(test_indices_tmp)];
    train_indices{f} = [train_indices{f}; class_k_indices(train_indices_tmp)];
  end  
end


% train and test
confusion_matrix = zeros(num_classes, num_classes);

options = [' -t 4 -s 0 -c ' num2str(config.C_one_vs_all)];

for f = 1 : num_fold
  model = train_1vsAll(Y(train_indices{f}), X(train_indices{f}, train_indices{f}), ...
      options, true);
  [Y_pred dec_values] = test_1vsAll(X(train_indices{f}, test_indices{f}), model);

  [confusion_matrix_tmp accuracy class_prec class_recall] = get_confusion_matrix(Y(test_indices{f}), Y_pred);
  %figure(f), imagesc(confusion_matrix_tmp), colormap gray; pause;
  confusion_matrix = confusion_matrix + confusion_matrix_tmp;
end

confusion_matrix = confusion_matrix / num_fold;
