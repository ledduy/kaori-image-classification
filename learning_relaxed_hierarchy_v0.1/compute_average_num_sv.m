function average_num_sv = compute_average_num_sv(mu, bin_model)

num_pos = sum(mu == 1);
num_neg = sum(mu == -1);

average_num_sv = num_neg/(num_neg + num_pos)*bin_model.totalSV/num_pos + ...
         num_pos/(num_neg + num_pos)*bin_model.totalSV/num_neg;