function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
K = num_labels;
labels_y = zeros(m, num_labels);
for c = 1:num_labels
  labels_y(:,c)=(y==c);
endfor
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
h = sigmoid(z3);
# labels_y : m,num_labels
# h : m,num_labels
tj = zeros(1, c);
tj1 = zeros(1, c);
for c = 1:K
  ty = labels_y(:,c);
  th = h(:,c);
  tj(:,c) = (ty' * log(th) + (1 - ty') * log(1 - th)) / -m;
endfor


t1_square_sum = sum(sum(Theta1(:,2:size(Theta1,2)) .^ 2));
t2_square_sum = sum(sum(Theta2(:,2:size(Theta2,2)) .^ 2));
J = sum(tj) + lambda * (t1_square_sum+t2_square_sum) / 2 / m;;




% -------------backpropagation---------------
# DELTA1 s2 * n+1
DELTA1 = zeros(size(Theta1));
#a2: m * s2, DELTA2 : k * s2
DELTA2 = zeros(size(Theta2));


sigmoidGradA2 = [zeros(size(z2,1), 1) sigmoidGradient(z2)];

for t=1:m
  # a3: 1 * k
  a3 = h(t, :);
  s2 = size(a2, 2);
  # delta3: 1 * k
  delta3 = a3 - labels_y(t, :);
  # Theta2: K * s2 , delta2: 1 * s2
  delta2 = delta3 * Theta2 .* sigmoidGradA2(t, :);
  delta2 = delta2(:, 2:end);

  DELTA2 = DELTA2 + delta3' * a2(t, :);
  DELTA1 = DELTA1 + delta2' * a1(t, :);
  
endfor
Theta1_grad = DELTA1 / m;
Theta2_grad = DELTA2 / m;

bias_Theta1_grad = Theta1_grad(:,1);
normal_bias_Theta1_grad = Theta1_grad(:,2:end)+lambda * Theta1(:,2:end) / m;
Theta1_grad = [bias_Theta1_grad normal_bias_Theta1_grad];

bias_Theta2_grad = Theta2_grad(:,1);
normal_bias_Theta2_grad = Theta2_grad(:,2:end)+lambda * Theta2(:,2:end) / m;
Theta2_grad = [bias_Theta2_grad normal_bias_Theta2_grad];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
