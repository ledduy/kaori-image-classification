function  mu_init = get_top_down_initialization(K, Y, alpha);

if size(Y, 1) < size(Y, 2)
   Y = Y'; 
end
% select the instances from active classes
y_space = unique(Y);
Y_new = Y;
for i = 1 : length(y_space)
    Y_new(Y==y_space(i)) = i;
end
Y = Y_new;

num_classes = length(y_space);

sel_inst_indx = [];
for i = 1 : num_classes
  pre_len = length(sel_inst_indx);
  sel_inst_indx = [sel_inst_indx; find(Y == i)];
  class_indx{i} = (pre_len+1 : length(sel_inst_indx))';
end

% normalized graph cut
K = double(K);
K_test = sparse(K(sel_inst_indx, sel_inst_indx));
d = sum(K_test, 2);
D = sparse(diag(d));
[V D1] = eigs(sqrt(D) * K_test * sqrt(D), 2);
w = V(:,2);

% label the instance
inst_mu = zeros(length(sel_inst_indx), 1);
inst_mu(w < 0) = -1;
inst_mu(w >= 0) = 1;

% label the class
% mu: -2 not reached by the node, 0 ignored, -1 neg, +1 pos
mu = zeros(num_classes, 1);
q_array = zeros(num_classes, 1);
for i = 1 : num_classes
  q = sum(inst_mu(class_indx{i})) / length(class_indx{i});
  q_array(i) = q;
  if q <= -1 + alpha
    mu(i) = -1;
  else 
    if q >= 1 - alpha
      mu(i) = 1;
    end
  end
end

if sum(mu == 1) == 0 || sum(mu == -1) == 0
    [tmp min_indx] = min(q_array);
    mu(min_indx) = -1;
    
    [tmp max_indx] = max(q_array);
    if max_indx == min_indx
      [val, indx] = sort(q_array, 'descend');
      max_indx = indx(2);
    end

    mu(max_indx) = 1;
end

mu_init{1} = mu;