%simple tailored lssvm

function [alpha,b,H] = simplelssvm(X,Y,type, gam,sig2, kernel_type) 
% 
% MATLAB implementation of the LS-SVM algorithm. %
%
% This implementation is quite straightforward, based on MATLAB's
% backslash matrix division (or PCG if available) and total kernel
% matrix construction. It has some extensions towards advanced
% techniques, especially applicable on small datasets (weighed
% LS-SVM, gamma-per-datapoint)
% Inputs    
%         X             : N x d matrix with the inputs of the training data
%         Y             : N x 1 vector with the outputs of the training data
%         type          : 'function estimation' ('f') or 'classifier' ('c')
%         kernel(*)     : by default 'RBF_kernel', can be RBF4_kernel;lin_kernel;poly_kernel
%         gam           : trade-off between the training error minimization and smoothness. 
%         sig2          : ?2 is the squared bandwidth
%
% CHECK TYPE
%
if type(1)~='f'
    if type(1)~='c'
        if type(1)~='t'
            if type(1)~='N'
                error('type has to be ''function (estimation)'', ''classification'', ''timeserie'' or ''NARX''');
            end
        end
    end
end

% check datapoints
x_dim = size(X,2);
y_dim = size(Y,2);

if and(type(1)~='t',and(size(X,1)~=size(Y,1),size(X,2)~=0)), error('number of datapoints not equal to number of targetpoints...'); end  
nb_data = size(X,1);%number of instance
%if size(X,1)<size(X,2), warning('less datapoints than dimension of a datapoint ?'); end
%if size(Y,1)<size(Y,2), warning('less targetpoints than dimension of a targetpoint ?'); end
if isempty(Y), error('empty datapoint vector...'); end

%
% initializing kernel type
%try kernel_type = kernel_type; catch, kernel_type = 'RBF_kernel'; end

%
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

%fprintf('~');
%
% is it weighted LS-SVM ?
%
weighted = (length(gam)>y_dim);
if and(weighted,length(gam)~=nb_data),
  warning('not enough gamma''s for Weighted LS-SVMs, simple LS-SVM applied');
  weighted=0;
end

% computation omega and H
omega = kernel_matrix(xtrain(selector, 1:x_dim), ...
    kernel_type, kernel_pars);


% initiate alpha and b
b = zeros(1,y_dim);
alpha = zeros(nb_data,y_dim);

for i=1:y_dim,
    H = omega;
    selector=~isnan(ytrain(:,i));
	%Indexes of training data effectively used during training
    nb_data=sum(selector);
    if size(gam,2)==nb_data, 
      try invgam = gam(i,:).^-1; catch, invgam = gam(1,:).^-1;end
      for t=1:nb_data, H(t,t) = H(t,t)+invgam(t); end
    else
      try invgam = gam(i,1).^-1; catch, invgam = gam(1,1).^-1;end
      for t=1:nb_data, H(t,t) = H(t,t)+invgam; end
    end    

    v = H(selector,selector)\ytrain(selector,i);
    %eval('v  = pcg(H,ytrain(selector,i), 100*eps,nb_data);','v = H\ytrain(selector, i);');
    nu = H(selector,selector)\ones(nb_data,1);
    %eval('nu = pcg(H,ones(nb_data,i), 100*eps,nb_data);','nu = H\ones(nb_data,i);');
    s = ones(1,nb_data)*nu(:,1);
    b(i) = (nu(:,1)'*ytrain(selector,i))./s;
    alpha(selector,i) = v(:,1)-(nu(:,1)*b(i));
end
return




