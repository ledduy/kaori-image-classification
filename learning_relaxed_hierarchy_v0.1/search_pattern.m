function [pattern_pool model_indx] = search_pattern(pattern_pool, pattern)

b_found = false;
for i = pattern_pool.end_indx : -1 :1
    dist = sum(abs(pattern - pattern_pool.patterns(:, i)));
    if dist == 0
      b_found = true;
      break;
    end
end

if b_found
  model_indx = pattern_pool.model_indices(i);
else
  model_indx = -1;
end