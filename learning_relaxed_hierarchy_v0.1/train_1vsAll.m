function model = train_1vsAll(Y, X, options, b_balance, varargin)
%%------------------------------------------------------
%                                   Training one-vs-all
%%------------------------------------------------------
disp(['start training 1vsAll model... ']);

num_classes = length(unique(Y));
model = cell(num_classes, 1);

options_new = cell(num_classes, 1);

parfor class_ind = 1:num_classes
    Y_binary = 2*(Y == class_ind)-1; %pos = 1, neg = -1
    
    num_pos = sum(Y_binary > 0);
    num_neg = sum(Y_binary < 0);
    
    if b_balance
      options_new{class_ind} = [options ' -w-1 1 -w1 ' num2str(num_neg/num_pos)];
    end
    
    model{class_ind} = svmtrain(double(Y_binary(:)), double([(1:length(Y_binary))' X]), ...
        options_new{class_ind});

    if Y_binary(1) < 0
        model{class_ind}.sv_coef  = - model{class_ind}.sv_coef ;
        model{class_ind}.rho      = - model{class_ind}.rho ;
    end
end

