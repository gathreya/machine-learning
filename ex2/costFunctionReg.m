function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = theta' * X';
h = sigmoid(z);
thetaLen = length(theta);
thetaWithout0 = theta(2:thetaLen,:);
hWithout0 = h(:, 2:m);
ywithout0 = y(2:m, :);
Xwithout0 = X(:, 2:thetaLen);

h0 = h(:, 1);
y0 = y(1, :);
X0 = X(:, 1);

gradient = zeros(size(theta));
J = (1/m * (-(y' * log(h)') - ((1-y)' * log(1-h)'))) + (lambda /(2 * m)) * sum(thetaWithout0.^2);
gradient = (1 / m) * (h - y') * X0;
firstPart = (1 / m) * ((h - y') * Xwithout0)
secondPart = (lambda / m) .* thetaWithout0
gradientRest =  firstPart + secondPart';

grad = [gradient, gradientRest];


% =============================================================

end
