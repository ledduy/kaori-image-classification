% This make.m is used under Windows

mex -O -c svm.cpp
%mex -O -c svm_model_matlab.c
mex -O -c svm_model_matlab.cpp
%mex -O svmtrain.c svm.obj svm_model_matlab.obj
mex -O svmtrain.cpp svm.o svm_model_matlab.o
%mex -O svmpredict.c svm.o svm_model_matlab.obj
mex -O svmpredict.cpp svm.o svm_model_matlab.o
mex -O libsvmread.cpp
mex -O libsvmwrite.cpp
%mex -O libsvmread.c
%mex -O libsvmwrite.c
