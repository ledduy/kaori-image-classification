% get the confusion matrix
% Author: Tianshi Gao <tianshig@stanford.edu>
% Date:   7/30/2010

function [conf_matrix overall_accuracy class_prec class_recall] = get_confusion_matrix(y, predLabel)

numClasses = max(y);
numInstance = length(y);
conf_matrix = zeros(numClasses, numClasses);
class_prec = zeros(numClasses, 1);
class_recall = zeros(numClasses, 1);
numCorrect = 0;
for i = 1 : length(predLabel)
   conf_matrix(y(i), predLabel(i)) = conf_matrix(y(i), predLabel(i)) + 1;
end

% compute class accuracy
for i = 1:numClasses
    if conf_matrix(i,i) == 0
        class_recall(i) = 0;
        class_prec(i) = 0;
    else
        class_recall(i) = conf_matrix(i, i) / sum(conf_matrix(i,:));
        class_prec(i) = conf_matrix(i, i) / sum(conf_matrix(:,i));
    end
end

for i = 1 : numClasses
    numCorrect = numCorrect + conf_matrix(i, i);
    n = sum(conf_matrix(i, :));
    conf_matrix(i, :) = conf_matrix(i, :) ./ n;
end

overall_accuracy = numCorrect / length(predLabel);



