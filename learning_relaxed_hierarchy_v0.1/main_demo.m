% -------------------------------------------
%                                   load data
% -------------------------------------------
train_fn = 'train_kernel__objectBank__split_01__rbf.mat';
test_fn = 'test_kernel__objectBank__split_01__rbf.mat';
load(train_fn);

% -------------------------------------------
%                            load config file
% -------------------------------------------
config_filename = 'config_demo_scene15.txt';
config = load_config_file(config_filename);


% -------------------------------------------
%                                    training
% -------------------------------------------
if ~isempty(config.confusion_matrix_fn)
  if exist(config.confusion_matrix_fn, 'file')
    load(config.confusion_matrix_fn);
  else
    conf_matrix = estimate_confusion_matrix(Y_train, K_train, config);  
    save('-v7.3', config.confusion_matrix_fn, 'conf_matrix');
  end
else
  conf_matrix = estimate_confusion_matrix(Y_train, K_train, config);  
end

config.confusion_matrix = conf_matrix;

model = relaxed_hierarchy_train(Y_train, K_train, config);

% -------------------------------------------
%                                     testing
% -------------------------------------------
clear Y_train;
clear K_train;
load(test_fn);
[Y_pred, accuracy, confusion_matrix, error_info, ...
 kernel_eval_cnt, classifier_eval_cnt] = relaxed_hierarchy_predict(Y_test, K_test, model, config.num_proc);