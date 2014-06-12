function kernel_indx = get_libsvm_kernel_index(kernel_type)

switch (kernel_type)
    case 'linear'           % linear
        kernel_indx = 0;
    case 'polynomial'       % polynomial
        kernel_indx = 1;
    case 'rbf'              % rbf
        kernel_indx = 2;
    case 'sigmoid'          % sigmoid
        kernel_indx = 3;
    case 'precomputed'      % rbf
        kernel_indx = 4;
    case 'hist_inter'       % hist_inter
        kernel_indx = 5;
    case 'chi-squre'        % chi-square
        kernel_indx = 6;
    otherwise
        kernel_indx = -1;
        disp('unknown kernel!');
        return;
end

