%% Training
clear all; close all; clc;
load 'fisheriris'
input_layers = 4;
hidden1 = 5;
hidden2 = 5;
output_layers = 3;

training_data = [];
desired_train_output = [];
w_h1 = rand(input_layers,hidden1);
w_h2 = rand(hidden1,hidden2);
w_h2o = rand(hidden2,output_layers);

alpha = 0.01; % learning rates

% Coding of 3 different classes
class1 = [1 0 0]';
class2 = [0 1 0]';
class3 = [0 0 1]';

for i=1:17  % 1:17 1:25 1:33 1:42
     template = [meas(i,:);meas(50+i,:);meas(100+i,:)]';
     training_data = [training_data template];
     desired_train_output = [desired_train_output,class1,class2,class3];
end
size_training = size(training_data);

max_iter = 20000;
error = zeros(output_layers,max_iter);

for iter=1:max_iter
        
        % Forward propagation 
        
        z1 = training_data'*w_h1;               % propagate from input to hidden layer 1
        h1 = sigmoid(z1);                       % output of hidden layer 1
        z2 = h1*w_h2;                           % propagate from hidden layer 1 to hidden layer 2
        h2 = sigmoid(z2);                       % output of hidden layer 2
        z3 = h2*w_h2o;                          % propagate from hidden layer 2 to output
        h3 = sigmoid(z3);                       % final output
        e = -(desired_train_output'-h3);              % output error
       
        
        % Back propagation
        
        delta_out = sig_derivative(h3).*e; 
        delta_h2 = sig_derivative(h2)'.*(w_h2o*delta_out'); 
        delta_h1 = sig_derivative(h1)'.*(w_h2*delta_h2);  
       
        w_h2o = w_h2o - alpha.*(h2'*delta_out);          % update ws between out hlayer and 2nd layer
        w_h2 = w_h2 - alpha.*(h1'*delta_h2');            % update ws between 2nd hlayer and 1st hlayer 
        w_h1 = w_h1 - alpha.*(training_data*delta_h1');  % update ws between 1st hlayer and in layer
        
        % Forward propagate for convergence verification
                
        z1 = training_data'*w_h1;               % propagate from input to hidden layer 1
        h1 = sigmoid(z1);                       % output of hidden layer 1
        z2 = h1*w_h2;                           % propagate from hidden layer 1 to hidden layer 2
        h2 = sigmoid(z2);                       % output of hidden layer 2
        z3 = h2*w_h2o;                          % propagate from hidden layer 2 to output
        h3 = sigmoid(z3);                       % final output
        e = -(desired_train_output'-h3);              % output error
        
        % Convergence verification
        if(norm(e)<2)
            break;
        end
        
        iter;
    
end

% Find misclassifications for training
out1 = [];
for i=1:size(training_data,2)
    z1 = training_data(:,i)'*w_h1;
    h1 = sigmoid(z1);
    z2 = h1*w_h2;                           
    h2 = sigmoid(z2);                       
    z3 = h2*w_h2o;                          
    out1(:,i) = sigmoid(z3);                       
end

out1(out1==max(out1))=1;       % Make greatest propability equal to 1
out1(out1~=1)=0;              % Make lowest probabilities equal to 0
linear_index = find(out1(:,:)~=desired_train_output(:,:));   % Find mismatches (linear indices)
s = size(out1);

% Find index of mismatch between output and desired test output

[I1,J1] = ind2sub(s,linear_index);
% disp(out1)
 disp('Misclassifications (Training): ');
disp(length(J1)/2) % since one column mismatch implies two of the elements in the column are
                  % mismatching, divide by two to get rid of repeated
                  % column mismatches 


%% Testing

 testing_data = [meas(18:50,:);meas(68:100,:);meas(118:150,:)]'; % Case 1, 99 test samples
% testing_data = [meas(26:50,:);meas(76:100,:);meas(126:150,:)]'; % Case 2, 75 test samples
% testing_data = [meas(34:50,:);meas(84:100,:);meas(134:150,:)]'; % Case 3, 51 test examples
% testing_data = [meas(43:50,:);meas(93:100,:);meas(143:150,:)]'; % Case 4, 24 test samples
 
size_test = size(testing_data);

desired_test_out = [repmat(class1,[1,max(size_test)/3]),...
    repmat(class2,[1,max(size_test)/3]),repmat(class3,[1,max(size_test)/3])];
out2 = [];
for i=1:size(testing_data,2)
    z1 = testing_data(:,i)'*w_h1;
    h1 = sigmoid(z1);
    z2 = h1*w_h2;                           
    h2 = sigmoid(z2);                       
    z3 = h2*w_h2o;                          
    out2(:,i) = sigmoid(z3);                       
end

% Find misclassifications for testing

out2(out2==max(out2))=1;       % Make greatest propability equal to 1
out2(out2~=1)=0;              % Make lowest probabilities equal to 0
linear_index = find(out2(:,:)~=desired_test_out(:,:));   % Find mismatches (linear indices)
s = size(out2);

% Find index of mismatch between output and desired test output

[I2,J2] = ind2sub(s,linear_index);
% disp(out2)
 disp('Misclassifications (Testing): ');
disp(length(J2)/2) % since one column mismatch implies two of the elements in the column are
                  % mismatching, divide by two to get rid of repeated
                  % column mismatches 

disp('Training Iterations for Convergence: ');
disp(iter)


%% Functions

function y = sigmoid(x)

y = 1.0 ./ (1 + exp(-x));

end

function yd = sig_derivative(x)

yd = x.*(1 - x);

end
