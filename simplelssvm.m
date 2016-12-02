%simple tailored lssvm

function [alpha,b,H] = simplelssvm(X,Y, gam,sig2, kernel_type) 
% 
% MATLAB implementation of the LS-SVM algorithm. %
%
% Inputs    
%         X             : N x d matrix with the inputs of the training data
%         Y             : N x 1 vector with the outputs of the training data
%         type          : 'function estimation' ('f') or 'classifier' ('c')
%         kernel(*)     : by default 'RBF_kernel', can be RBF4_kernel;lin_kernel;poly_kernel
%         gam           : trade-off between the training error minimization and smoothness. 
%         sig2          : ?2 is the squared bandwidth
%
% reference: Suykens, J. A. K., et al. "Least squares support vector machine classifiers: a large scale algorithm." European Conference on Circuit Theory and Design, ECCTD. Vol. 99. 1999.


% check datapoints
x_dim = size(X,2);
y_dim = size(Y,2);

nb_data = size(X,1);%number of instance

if isempty(Y), error('empty datapoint vector...'); end

% initiate datapoint selector
xtrain = X;
ytrain = Y;
selector=1:nb_data; %Indexes of training data effectively used during training

%
% regularisation term and kenel parameters
if(gam<=0), error('gam must be larger then 0');end
%
% initializing kernel type
try kernel_type = kernel_type; catch, kernel_type = 'RBF_kernel';end
if sig2<=0,
  kernel_pars = (x_dim);
else
  kernel_pars = sig2;
end

% computation omega and H
omega = kernel_matrix(xtrain(selector, 1:x_dim), ...
    kernel_type, kernel_pars);


% initiate alpha and b
b = zeros(1,y_dim);
alpha = zeros(nb_data,y_dim);

%initiate v and nu
v=zeros(nb_data,1);
nu=zeros(nb_data,1);
H = omega;
[v]= conjgrad(H,ytrain,v)
[nu]= conjgrad(H,ones(nb_data,1),nu)
s = ones(1,nb_data)*nu;
b= (nu'*ytrain)./s;
alpha= v-(nu*b);

    
% for i=1:y_dim,
%     H = omega;
%     selector=~isnan(ytrain(:,i));
% 	%Indexes of training data effectively used during training
%     nb_data=sum(selector);
%     if size(gam,2)==nb_data, 
%       try invgam = gam(i,:).^-1; catch, invgam = gam(1,:).^-1;end
%       for t=1:nb_data, H(t,t) = H(t,t)+invgam(t); end
%     else
%       try invgam = gam(i,1).^-1; catch, invgam = gam(1,1).^-1;end
%       for t=1:nb_data, H(t,t) = H(t,t)+invgam; end
%     end    
%     v = H(selector,selector)\ytrain(selector,i);
%     nu = H(selector,selector)\ones(nb_data,1);
%     s = ones(1,nb_data)*nu(:,1);
%     b(i) = (nu(:,1)'*ytrain(selector,i))./s;
%     alpha(selector,i) = v(:,1)-(nu(:,1)*b(i));
% end

return




