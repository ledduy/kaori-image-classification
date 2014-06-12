function model = my_train_svm(Y, X, config)
% Y is binary

% set options
options = [' -q -t 4 -s 0 -c ' num2str(config.C)];

% deal with unbalanced data
num_pos = sum(Y == 1);
num_neg = sum(Y == -1);

% deal with unbalanced set if the ratio is larger than 2
ratio = num_pos / num_neg;
if ratio >= 2
  options = sprintf('%s -w-1 %f -w1 1', options, ratio);
elseif 1/ratio >=2
  options = sprintf('%s -w-1 1 -w1 %f ', options, 1/ratio);
end

% train svm
model = svmtrain(double(Y), [(1:length(Y))' double(X)], options);

% libsvm always treat the first instance as "positive"
if Y(1) < 0
    model.sv_coef  = -model.sv_coef ;
    model.rho      = -model.rho ;
end

