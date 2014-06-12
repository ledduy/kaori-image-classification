function [pattern_pool] = add_pattern(pattern_pool, pattern, model_indx)

[m, n] = size(pattern_pool.patterns);
if pattern_pool.end_indx+1 > n % expand
    new_patterns = sparse(m, 2*n);
    new_patterns(:, 1:n) = pattern_pool.patterns;
    pattern_pool.patterns = new_patterns;
    clear new_patterns;
    
    new_indices = zeros(1, 2*n);
    new_indices(1:n) = pattern_pool.model_indices;
    pattern_pool.model_indices = new_indices;
    clear new_indices;
end

pattern_pool.end_indx = pattern_pool.end_indx + 1;
pattern_pool.patterns(:, pattern_pool.end_indx) = pattern;
pattern_pool.model_indices(pattern_pool.end_indx) = model_indx;



