function  [mu model] = train_node_model_with_initialization(Y, X, mu, config)
% assume that the index of Y is from 1 to K (num_classes)

N = size(X, 1); % # of instances 
num_classes = length(unique(Y));

% check the dimension
assert(length(Y) == N, 'error: the number of labels does not match the number of instances!');
assert(length(mu) == num_classes, 'error: the dimension of mu is not equal to the number of classes');

% alternating method
iter = 1;
while(1)
    % step 1: given the coloring mu, train the binary classifier
    Y_color = mu(Y);
    train_indx = find(Y_color ~= 0);
    model = my_train_svm(double(Y_color(train_indx)), double(X(train_indx, train_indx)), config);
    
    if (iter == config.max_num_iter)
      break;
    end
    
    % step 2: given the binary classifier, train the coloring mu
    prev_mu = mu;
    mu = color_label_space(Y, X, model, config);
    
    % stop: if reached the maximum number of iterations
    %       or converged
    iter = iter + 1;
    if (norm(prev_mu - mu) == 0)
        break;
    end
end
