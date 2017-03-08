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
theta_proxy = theta ;
theta_proxy(1) = 0 ; %setting first element to zero to exclude theta_o from regulations
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

ht = (sigmoid(X*theta)) ;

J = (-1.0/m)* (y'*log(ht) + (1-y)'*log(1-ht)) + (lambda /2.0/m) * theta_proxy' * theta_proxy ;

grad = X'*(ht -y)/m + lambda/m * theta_proxy ;


% =============================================================

end
