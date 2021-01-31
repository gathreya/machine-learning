function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

C = 1;
sigma = 1;
errors = zeros(length(Cvalues) * length(sigmaValues), 1);
error_temp = 100;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using

i = 0;
for Cvalue = Cvalues
    for sigmaValue = sigmaValues
        i = i+1;
        model= svmTrain(X, y, Cvalue, @(x1, x2) gaussianKernel(x1, x2, sigmaValue));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval)); 
        if (error < error_temp) 
            error_temp = error;
            C = Cvalue;
            sigma = sigmaValue;
        end
    end
end
%

fprintf('C is %f', C);
fprintf('sigma is %f', sigma);






% =========================================================================

end
