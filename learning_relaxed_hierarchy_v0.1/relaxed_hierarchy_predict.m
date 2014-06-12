function [Y_pred, accuracy, confusion_matrix, ...
         error_info, kernel_eval_cnt, classifier_eval_cnt] = relaxed_hierarchy_predict(Y, X, model, varargin)

num_instance = size(X, 2);
error_info = cell(num_instance, 1);
kernel_eval_cnt = zeros(num_instance, 1);
classifier_eval_cnt = zeros(num_instance, 1);
Y_pred = zeros(num_instance, 1);


internal_error_cnt = 0;
leaf_error_cnt = 0;

num_proc = 1;
if size(varargin, 2) ~= 0
  num_proc = varargin{1};
end

if num_proc > 1
  matlabpool(num_proc);
end

% one instance at a time
parfor i = 1 : num_instance
    disp(['processing ' num2str(i) '/' num2str(num_instance) '...']);
    [Y_pred(i), error_info{i}, kernel_eval_cnt(i), classifier_eval_cnt(i)] = ...
        relaxed_hierarchy_predict_single(Y(i), X(:,i), model);
    
    if error_info{i}.has_error
      if error_info{i}.has_internal_error
        internal_error_cnt = internal_error_cnt + 1;
      else
        leaf_error_cnt = leaf_error_cnt + 1;
      end
    end    
end

if num_proc > 1
  matlabpool close;
end

if size(Y, 1) < size(Y,2)
   Y = Y'; 
end
total_error_cnt = sum(Y_pred ~= Y);
assert(internal_error_cnt + leaf_error_cnt == total_error_cnt);

% get error info
num_level = model.node_models{end}.level;
error_level = zeros(num_level, 1);

for i = 1 : num_instance
  if error_info{i}.has_error
    error_level(error_info{i}.level) = error_level(error_info{i}.level) + 1;
  end
end

[confusion_matrix accuracy class_prec class_recall] = get_confusion_matrix(Y, Y_pred);

disp(['overall accuracy: ' num2str(accuracy)]);
disp(['internal node error rate: ' num2str(internal_error_cnt/num_instance)]);
disp(['leaf node (1vsAll) error rate: ' num2str(leaf_error_cnt/num_instance)]);
disp(['mean number of classifier evaluations ' num2str(mean(classifier_eval_cnt))]);
disp(['mean number of kernel evaluations ' num2str(mean(kernel_eval_cnt))]);
for i = 1 : num_level
  disp(['level ' num2str(i) ' has ' num2str(error_level(i)) ' errors']);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y_pred, error_info, ...
          kernel_eval_cnt, classifier_eval_cnt] = ...
         relaxed_hierarchy_predict_single(Y, X, model)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% one instance at a time

node_indx = 1;
union_sv_indice = [];
classifier_eval_cnt = 0;

error_info.has_error = false;
error_info.level = -1;
error_info.node = -1;
error_info.has_internal_error = false;

Y_pred = -1;

while (1)   
  binary_model = model.node_models{node_indx}.binary_model;

  if length(binary_model) > 1 % 1vsAll node
    num_active_classes = length(binary_model);
    resp = zeros(num_active_classes, 1);
    [Y_pred_leaf dec_values_leaf] = test_1vsAll(X, binary_model, true);
    
    for k = 1 : num_active_classes
      % record kernel evaluations
      union_sv_indice = union(union_sv_indice,...
                              binary_model{k}.global_SV_indice);
    end
    
    % record classifier evaluations
    classifier_eval_cnt = classifier_eval_cnt + num_active_classes;

    mu = model.node_models{node_indx}.mu;
    active_classes_indx = find(mu ~= -2);
    Y_pred = active_classes_indx(Y_pred_leaf);
    
    if Y_pred ~= Y && ...
       error_info.has_internal_error == false && ...
       Y ~= -1 

      error_info.has_error = true;
      error_info.level = model.node_models{node_indx}.level;
      error_info.node = node_indx;
    end
    kernel_eval_cnt = length(union_sv_indice);
    return;
    
  else % internal nodes

    % the last argument "true" is to use the global indx of SVs
    resp = eval_binary_model_K(X, binary_model, true); 
    classifier_eval_cnt = classifier_eval_cnt + 1;
    union_sv_indice = union(union_sv_indice, ...
                            binary_model.global_SV_indice);

    mu = model.node_models{node_indx}.mu;
    if resp < 0
      if Y ~= -1 
        if mu(Y) == 1 && error_info.has_error == false
          error_info.has_error = true;
          error_info.has_internal_error = true;
          error_info.level = model.node_models{node_indx}.level;
          error_info.node = node_indx;
        end
      end
      
      if isempty(model.node_models{node_indx}.left_indx)
        assert(sum(mu == -1) == 1);
        assert(sum(mu == 0) == 0);

        Y_pred = find(mu == -1);
        kernel_eval_cnt = length(union_sv_indice);
        return;
      else
        node_indx = model.node_models{node_indx}.left_indx;
      end
    else
      if Y ~= -1
        if mu(Y) == -1 && error_info.has_error == false
          error_info.has_error = true;
          error_info.has_internal_error = true;
          error_info.level = model.node_models{node_indx}.level;
          error_info.node = node_indx;
        end
      end
      if isempty(model.node_models{node_indx}.right_indx)
        assert(sum(mu == 1) == 1);
        assert(sum(mu == 0) == 0);
        
        Y_pred = find(mu == 1);
        kernel_eval_cnt = length(union_sv_indice);
        return;
      else
        node_indx = model.node_models{node_indx}.right_indx;
      end
    end
  end
end

