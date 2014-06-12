function model = train_hierarchical_part(Y, X, config)

num_classes = length(unique(Y));

pattern_pool.patterns = sparse(num_classes, num_classes*num_classes);
pattern_pool.model_indices = zeros(1, num_classes*num_classes);
pattern_pool.end_indx = 0;

% K is n-by-n
global_indice = [1:size(X, 1)]';
model = [];

% set confusion_matrix
confusion_matrix = config.confusion_matrix;

% training the first stage with sharing
node_cnt = 0;
for l = 1 : config.hierarchy_level
    disp(['---------- training level ' num2str(l) ' -------------']);
    level_node_cnt = 0;
    tstart_level = tic;
    
    if l == 1
        node_cnt = node_cnt + 1;
        level_node_cnt = level_node_cnt + 1;
        disp(['training node ' num2str(node_cnt) '...']);
        node_models{node_cnt} = train_node_model(Y, X, confusion_matrix, config);
        node_models{node_cnt}.level = l;
        node_models{node_cnt}.right_indx = [];
        node_models{node_cnt}.left_indx = [];
    else
        flag1 = false;
        flag2 = false;
        for k = 1 : length(node_models)
            if (node_models{k}.level ~= l - 1)
                continue;
            end

            %% left node (active classes are those labeled as -1 or 0)
            active_classes_binary = (node_models{k}.mu <= 0) & ...
                (node_models{k}.mu > -2); % -2 indicates
                                          % classes that have
                                          % been pruned away
            active_classes = find(active_classes_binary == 1);
            if length(active_classes) >= 2
                % search whether there is a node with the same
                % active classes
                active_class_pattern = double(active_classes_binary);
                [pattern_pool model_indx] = search_pattern(pattern_pool, active_class_pattern);
                if model_indx > 0
                    disp('merge node...');
                    node_models{k}.left_indx = model_indx;
                else % add the pattern and train the node
                    level_node_cnt = level_node_cnt + 1;
                    node_cnt = node_cnt + 1;
                    pattern_pool = add_pattern(pattern_pool, active_class_pattern, node_cnt);
                    train_indx = [];
                    for i = 1 : num_classes
                        if node_models{k}.mu(i) <= 0 && ...
                              node_models{k}.mu(i) > -2
                            train_indx = [train_indx; find(Y == i)];
                        end
                    end
                    disp(['training node ' num2str(node_cnt) '...']);
                    node_models{node_cnt} = train_node_model(Y(train_indx), X(train_indx, :), ...
                                                             confusion_matrix(active_classes, active_classes), config);
                    % update the mu
                    node_models{node_cnt}.mu = update_mu(active_classes, ...
                                                         num_classes,...
                                                         node_models{node_cnt}.mu);
                    
                    node_models{node_cnt}.right_indx = [];
                    node_models{node_cnt}.left_indx = [];
                    node_models{k}.left_indx = node_cnt;
                    node_models{node_cnt}.level = l;
                end
                flag1 = true;
            end
            
            %% right node (active classes are those labeled as 1 or 0)
            active_classes = find(node_models{k}.mu >= 0);
            if length(active_classes) >= 2 
                % search whether there is a node with the same active classes
                active_class_pattern = double(node_models{k}.mu >= 0);
                [pattern_pool model_indx] = search_pattern(pattern_pool, active_class_pattern);
                if model_indx > 0
                    disp('merge node...');
                    node_models{k}.right_indx = model_indx;
                else % add the pattern and train the node
                    level_node_cnt = level_node_cnt + 1;
                    node_cnt = node_cnt + 1;
                    pattern_pool = add_pattern(pattern_pool, active_class_pattern, node_cnt);
                    train_indx = [];
                    for i = 1 : num_classes
                        if node_models{k}.mu(i) >= 0
                            train_indx = [train_indx; find(Y == i)];
                        end
                    end
                    disp(['training node ' num2str(node_cnt) '...']);
                    node_models{node_cnt} = train_node_model(Y(train_indx), X(train_indx, :), ...
                                                             confusion_matrix(active_classes, active_classes), config);
                    % update the mu
                    node_models{node_cnt}.mu = update_mu(active_classes, ...
                                                         num_classes,...
                                                         node_models{node_cnt}.mu);
                    
                    node_models{node_cnt}.right_indx = [];
                    node_models{node_cnt}.left_indx = [];
                    node_models{k}.right_indx = node_cnt;
                    node_models{node_cnt}.level = l;
                end
                flag2 = true;
            end
            
        end
        
        telapsed_level = toc(tstart_level);
        fprintf('level %d has %d nodes, training takes %f (s)\n\n', ...
            l, level_node_cnt, telapsed_level);
        if flag1 == false && flag2 == false
            break;
        end
    end
end

model.config = config;
model.node_models = node_models;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mu = update_mu(active_classes, total_num_classes, mu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu_new = -2 * ones(total_num_classes, 1);

assert(length(active_classes) == length(mu));
for i = 1 : length(mu)
  mu_new(active_classes(i)) = mu(i);
end

mu = mu_new;

