function mu = get_bottom_up_initialization(confusion_matrix, num_pairs)

% symmetrize confusion matrix
confusion_matrix = confusion_matrix' + confusion_matrix;
num_classes = size(confusion_matrix, 1);

% get the pairs of classes with minimum confusions
[min_conf_val min_conf_indx] = sort(confusion_matrix(:));

% sample pair of classes with little confusion
max_num_pairs = num_classes*(num_classes-1)/2;
num_pairs = min(num_pairs, max_num_pairs);
mu = {};
for i = 1 : num_pairs
  [min_row_indx min_col_indx] = ind2sub(size(confusion_matrix), min_conf_indx(i));
  if min_row_indx == min_col_indx
    continue;
  end

  mu{i} = zeros(num_classes, 1);
  mu{i}(min_col_indx) = 1;
  mu{i}(min_row_indx) = -1;
end