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

temp = 0;
tlambda = 0;
for i = 1:m
	temp += (-y(i)*log(sigmoid(theta'*X'(:,i)))) - ((1-y(i))*log(1-sigmoid(theta'*X'(:,i))));
end

for i = 2:size(theta,1)
  tlambda += pow2(theta(i));
end
J = (temp/m) + ((lambda*tlambda)/(2*m));

temp = zeros(size(theta(2:end,:))(1),1);
ftheta = 0;
for i = 1:m
	 ftheta += (sigmoid(theta'*X'(:,i))-y(i))*X'(1,i);
end

for i = 1:m
	temp += ((sigmoid(theta'*X'(:,i))-y(i))*X'(2:end,i)) + (lambda/m)*theta(2:end,1);
end
temp = [ftheta;temp];
grad = temp/m;

% =============================================================

end
