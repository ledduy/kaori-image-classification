function mu = color_label_space(Y, X, model, config)

y_space = unique(Y);
num_classes = length(y_space);
mu = zeros(num_classes, 1);

resp = eval_binary_model_K(X, model);

xi_pos_class = zeros(num_classes, 1);
xi_neg_class = zeros(num_classes, 1);

delta_pos = zeros(num_classes, 1);
delta_neg = zeros(num_classes, 1);

for i = 1 : num_classes
    n = length(find(Y==y_space(i))); % number of instances from
                                     % class i

    xi_pos_class(i) = sum(max([zeros(n, 1), 1-resp(Y == y_space(i))], [], 2));
    xi_neg_class(i) = sum(max([zeros(n, 1), 1+resp(Y == y_space(i))], [], 2));
    
    delta_pos(i) = 1/n*xi_pos_class(i) - config.rho;
    delta_neg(i) = 1/n*xi_neg_class(i) - config.rho;
    
    if delta_pos(i) > 0 && delta_neg(i) > 0
      mu(i) = 0;
    else
      if (delta_pos(i) < delta_neg(i))
        mu(i) = 1;
      else
        mu(i) = -1;
      end
    end
end

%% test: enforce balance constraints
B = config.B;
B_prime = sum(mu);
if B_prime > B
  indx_zero = find(mu == 0);
  indx_pos = find(mu > 0);
  
  num_zero = length(indx_zero);
  num_pos = length(indx_pos);
  
  class_indx = zeros(num_zero+2*num_pos, 1);
  class_indx(1:num_zero) = indx_zero;
  class_indx(num_zero+1:end) = reshape([indx_pos, indx_pos]', num_pos*2,1);
  
  orig_mu = zeros(num_zero+2*num_pos, 1);
  orig_mu(1:num_zero) = 0;
  orig_mu(num_zero+1:end) = 1;
  
  delta_steps = zeros(num_zero+2*num_pos, 1);
  delta_steps(1:num_zero) = 1;
  
  new_mu = zeros(num_zero+2*num_pos, 1);
  new_mu(1:num_zero) = -1;
  
  S_delta = zeros(num_zero+2*num_pos, 1);
  S_delta(1:num_zero) = delta_neg(indx_zero);

  for i = 1 : num_pos
    delta_k = delta_neg(indx_pos(i));
    delta_k_0 = -delta_pos(indx_pos(i));

    tmp_indx = num_zero + (i-1)*2 + 1;
    if delta_k_0 <= delta_k
      new_mu(tmp_indx) = 0;
      new_mu(tmp_indx+1) = -1;
      
      S_delta(tmp_indx) = delta_k_0;
      S_delta(tmp_indx+1) = delta_k;
      
      delta_steps(tmp_indx) = 1;
      delta_steps(tmp_indx+1) = 1;
    else
      new_mu(tmp_indx) = -1;
      new_mu(tmp_indx+1) = 0;
      
      S_delta(tmp_indx) = (delta_k_0+delta_k)/2;
      S_delta(tmp_indx+1) = delta_k_0;
      
      delta_steps(tmp_indx) = 2;
      delta_steps(tmp_indx+1) = 1;
    end
  end

  [S_delta_sorted sorted_indx] = sort(S_delta);
  new_mu = new_mu(sorted_indx);
  orig_mu = orig_mu(sorted_indx);
  class_indx = class_indx(sorted_indx);
  valid_flag = ones(length(S_delta), 1);
  delta_steps = delta_steps(sorted_indx);
  
  d = 0;
  ctr = 0;
  while(1)
    if d == B_prime - B || d == B_prime - B + 1
      break;
    end

    while(1)
      ctr = ctr + 1;
      if valid_flag(ctr)
        break;
      end
    end

    if delta_steps(ctr) == 1
      mu(class_indx(ctr)) = new_mu(ctr);
      d = d+1;
    else
      % this case should happen only when rho >= 1  assert(config.rho >= 1);
      if d <= B_prime - B - 2
        mu(class_indx(ctr)) = new_mu(ctr);
        assert(new_mu(ctr) == -1); % TODO: remove later
        d = d + 2;
        valid_flag(find(class_indx == class_indx(ctr))) = 0;
      else
        Delta_k_minus = 2*S_delta_sorted(ctr);
        % find the next smallest Delta_j or Delta_{j,0}
        for itr = ctr + 1 : length(S_delta_sorted)
          if valid_flag(itr) == 0
            continue;
          end

          if delta_steps(itr) == 1
            j = itr;
            Delta_j_min = S_delta_sorted(j);
          end
        end
        
        % find the largest Delta_i or Delta_{i,0}
        Delta_i_max = 0;
        i = -1;
        for itr = ctr-1:-1:1
          if delta_steps(itr) == 1 && valid_flag(itr) == 1
            i = itr;
            Delta_i_max = S_delta_sorted(i);
          end
        end
        
        % find the l with the largest Delta_l_minus - Delta_l_0
        Delta_l_max = -inf;
        l = -1;
        for itr = ctr-1:-1:1
          if delta_steps(itr) == 2
            delta_tmp = xi_neg_class(class_indx(itr));
            if delta_tmp > Delta_l_max
              l = itr;
              Delta_l_max = delta_tmp;
            end
          end
        end
        
        % one-step-min = j
        if Delta_j_min <= Delta_k_minus - Delta_i_max && ...
              Delta_j_min <= Delta_k_minus - Delta_l_max
          mu(class_indx(j)) = new_mu(j);
          d = d + 1;

        else
          % one-step-min = Delta_k_minus - Delta_i_max
          if Delta_k_minus - Delta_i_max <= Delta_j_min && ...
                Delta_k_minus - Delta_i_max <= Delta_k_minus - Delta_l_max
            mu(class_indx(ctr)) = -1;
            if i > 0
              mu(class_indx(i)) = orig_mu(i);
              d = d + 1;
            else
              d = d + 2;
            end
          else
            assert(l > 0);
            mu(class_indx(l)) = 0;
            mu(class_indx(ctr)) = -1;
            d = d + 1;
          end
        end
      end
    end
  end
end

if B_prime < -B
  indx_zero = find(mu == 0);
  indx_neg = find(mu < 0);
  
  num_zero = length(indx_zero);
  num_neg = length(indx_neg);
  
  class_indx = zeros(num_zero+2*num_neg, 1);
  class_indx(1:num_zero) = indx_zero;
  class_indx(num_zero+1:end) = reshape([indx_neg, indx_neg]', num_neg*2,1);
  
  orig_mu = zeros(num_zero+2*num_neg, 1);
  orig_mu(1:num_zero) = 0;
  orig_mu(num_zero+1:end) = -1;
  
  delta_steps = zeros(num_zero+2*num_neg, 1);
  delta_steps(1:num_zero) = 1;
  
  new_mu = zeros(num_zero+2*num_neg, 1);
  new_mu(1:num_zero) = 1;
  
  S_delta = zeros(num_zero+2*num_neg, 1);
  S_delta(1:num_zero) = delta_pos(indx_zero);

  for i = 1 : num_neg
    delta_k = delta_pos(indx_neg(i));
    delta_k_0 = -delta_neg(indx_neg(i));

    tmp_indx = num_zero + (i-1)*2 + 1;
    if delta_k_0 <= delta_k
      new_mu(tmp_indx) = 0;
      new_mu(tmp_indx+1) = 1;
      
      S_delta(tmp_indx) = delta_k_0;
      S_delta(tmp_indx+1) = delta_k;
      
      delta_steps(tmp_indx) = 1;
      delta_steps(tmp_indx+1) = 1;
    else
      new_mu(tmp_indx) = 1;
      new_mu(tmp_indx+1) = 0;
      
      S_delta(tmp_indx) = (delta_k_0+delta_k)/2;
      S_delta(tmp_indx+1) = delta_k_0;
      
      delta_steps(tmp_indx) = 2;
      delta_steps(tmp_indx+1) = 1;
    end
  end

  [S_delta_sorted sorted_indx] = sort(S_delta);
  new_mu = new_mu(sorted_indx);
  orig_mu = orig_mu(sorted_indx);
  class_indx = class_indx(sorted_indx);
  valid_flag = ones(length(S_delta), 1);
  delta_steps = delta_steps(sorted_indx);
  
  d = 0;
  ctr = 0;
  while(1)
    if d == -B - B_prime || d == -B - B_prime + 1
      break;
    end

    while(1)
      ctr = ctr + 1;
      if valid_flag(ctr)
        break;
      end
    end

    if delta_steps(ctr) == 1
      mu(class_indx(ctr)) = new_mu(ctr);
      d = d+1;
    else
      % this case should happen only when rho >= 1  assert(config.rho >= 1);
      if d >= -B - B_prime  - 2
        mu(class_indx(ctr)) = new_mu(ctr);
        assert(new_mu(ctr) == 1); % TODO: remove later
        d = d + 2;
        valid_flag(find(class_indx == class_indx(ctr))) = 0;
      else
        Delta_k_minus = 2*S_delta_sorted(ctr);
        % find the next smallest Delta_j or Delta_{j,0}
        for itr = ctr + 1 : length(S_delta_sorted)
          if valid_flag(itr) == 0
            continue;
          end

          if delta_steps(itr) == 1
            j = itr;
            Delta_j_min = S_delta_sorted(j);
          end
        end
        
        % find the largest Delta_i or Delta_{i,0}
        Delta_i_max = 0;
        i = -1;
        for itr = ctr-1:-1:1
          if delta_steps(itr) == 1 && valid_flag(itr) == 1
            i = itr;
            Delta_i_max = S_delta_sorted(i);
          end
        end
        
        % find the l with the largest Delta_l_minus - Delta_l_0
        Delta_l_max = -inf;
        l = -1;
        for itr = ctr-1:-1:1
          if delta_steps(itr) == 2
            delta_tmp = xi_neg_class(class_indx(itr));
            if delta_tmp > Delta_l_max
              l = itr;
              Delta_l_max = delta_tmp;
            end
          end
        end
        
        % one-step-min = j
        if Delta_j_min <= Delta_k_minus - Delta_i_max && ...
              Delta_j_min <= Delta_k_minus - Delta_l_max
          mu(class_indx(j)) = new_mu(j);
          d = d + 1;

        else
          % one-step-min = Delta_k_minus - Delta_i_max
          if Delta_k_minus - Delta_i_max <= Delta_j_min && ...
                Delta_k_minus - Delta_i_max <= Delta_k_minus - Delta_l_max
            mu(class_indx(ctr)) = -1;
            if i > 0
              mu(class_indx(i)) = orig_mu(i);
              d = d + 1;
            else
              d = d + 2;
            end
          else
            assert(l > 0);
            mu(class_indx(l)) = 0;
            mu(class_indx(ctr)) = -1;
            d = d + 1;
          end
        end
      end
    end
  end
end


if any(mu == 1) == false % no positive class
    [min_val min_indx] = min(xi_pos_class);
    mu(min_indx) = 1;
end

if any(mu == -1) == false % no negative class
    [min_val min_indx] = min(xi_neg_class);
    mu(min_indx) = -1;
end