%my own tailored lssvm
function model = initlssvm(X,Y,type, gam,sig2, kernel_type)
% Initiate the object oriented structure representing the LS-SVM model
%
%   model = initlssvm(X,Y, type, gam, sig2)
%   model = initlssvm(X,Y, type, gam, sig2, kernel_type)
%
% Full syntax
% 
% >> model = initlssvm(X, Y, type, gam, sig2, kernel, preprocess)
              %gam=Regularisation parameter
% 
%       Outputs    
%         model         : Object oriented representation of the LS-SVM model
%       Inputs    
%         X             : N x d matrix with the inputs of the training data
%         Y             : N x 1 vector with the outputs of the training data
%         type          : 'function estimation' ('f') or 'classifier' ('c')
%         kernel(*)     : Kernel type (by default
%         'RBF_kernel')RBF4_kernel;lin_kernel;poly_kernel
%
% see also:
%   trainlssvm, simlssvm, changelssvm, codelssvm, prelssvm

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
%ouput 1
model.type = type;

%
% check datapoints
%output 2
model.x_dim = size(X,2);
model.y_dim = size(Y,2);

if and(type(1)~='t',and(size(X,1)~=size(Y,1),size(X,2)~=0)), error('number of datapoints not equal to number of targetpoints...'); end  
model.nb_data = size(X,1);%number of instance
%if size(X,1)<size(X,2), warning('less datapoints than dimension of a datapoint ?'); end
%if size(Y,1)<size(Y,2), warning('less targetpoints than dimension of a targetpoint ?'); end
if isempty(Y), error('empty datapoint vector...'); end

%
% initializing kernel type
%output 3
try model.kernel_type = kernel_type; catch, model.kernel_type = 'RBF_kernel'; end

%
% initiate datapoint selector
%output 6
model.xtrain = X;
model.ytrain = Y;
%output 7
model.selector=1:model.nb_data; %Indexes of training data effectively used during training

%
% regularisation term and kenel parameters
%output 8
if(gam<=0), error('gam must be larger then 0');end
model.gam = gam;

%
% initializing kernel type
%output 9
try model.kernel_type = kernel_type; catch, model.kernel_type = 'RBF_kernel';end
if sig2<=0,
  model.kernel_pars = (model.x_dim);
else
  model.kernel_pars = sig2;
end

%
% dynamic models
%output 10
model.x_delays = 0;
model.y_delays = 0;
model.steps = 1;

% for classification: one is interested in the latent variables or
% in the class labels
%output 11
model.latent = 'no';
%settings for weight function
%output 14
model.weights = [];

function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

% Construct the positive (semi-) definite and symmetric kernel matrix
%
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
%
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
%
% >> Omega_Xt = kernel_matrix(X, kernel_fct, sig2, Xt)
%
%
% Full syntax
%
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
% >> Omega = kernel_matrix(X, kernel_fct, sig2, Xt)
%
% Outputs
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs
%   X      : N x d matrix with the inputs of the training data
%   kernel : Kernel type (by default 'RBF_kernel')
%   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   Xt(*)  : Nt x d matrix with the inputs of the test data
%
% See also:
%  RBF_kernel, lin_kernel, kpca, trainlssvm, kentropy

[nb_data,d] = size(Xtrain);


if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(2*kernel_pars(1)));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./(2*kernel_pars(1)));
    end
    
elseif strcmp(kernel_type,'RBF4_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
end

function [model,H] = lssvmMATLAB(model) 
% 
% MATLAB implementation of the LS-SVM algorithm. %
%
% This implementation is quite straightforward, based on MATLAB's
% backslash matrix division (or PCG if available) and total kernel
% matrix construction. It has some extensions towards advanced
% techniques, especially applicable on small datasets (weighed
% LS-SVM, gamma-per-datapoint)
% gam: for gam low minimizing of the complexity of the model is emphasized, for gam high, fitting of the training data points is stressed.


%fprintf('~');
%
% is it weighted LS-SVM ?
%
weighted = (length(model.gam)>model.y_dim);
if and(weighted,length(model.gam)~=model.nb_data),
  warning('not enough gamma''s for Weighted LS-SVMs, simple LS-SVM applied');
  weighted=0;
end

% computation omega and H
omega = kernel_matrix(model.xtrain(model.selector, 1:model.x_dim), ...
    model.kernel_type, model.kernel_pars);


% initiate alpha and b
model.b = zeros(1,model.y_dim);
model.alpha = zeros(model.nb_data,model.y_dim);

for i=1:model.y_dim,
    H = omega;
    model.selector=~isnan(model.ytrain(:,i));
	%Indexes of training data effectively used during training
    nb_data=sum(model.selector);
    if size(model.gam,2)==model.nb_data, 
      try invgam = model.gam(i,:).^-1; catch, invgam = model.gam(1,:).^-1;end
      for t=1:model.nb_data, H(t,t) = H(t,t)+invgam(t); end
    else
      try invgam = model.gam(i,1).^-1; catch, invgam = model.gam(1,1).^-1;end
      for t=1:model.nb_data, H(t,t) = H(t,t)+invgam; end
    end    

    v = H(model.selector,model.selector)\model.ytrain(model.selector,i);
    %eval('v  = pcg(H,model.ytrain(model.selector,i), 100*eps,model.nb_data);','v = H\model.ytrain(model.selector, i);');
    nu = H(model.selector,model.selector)\ones(nb_data,1);
    %eval('nu = pcg(H,ones(model.nb_data,i), 100*eps,model.nb_data);','nu = H\ones(model.nb_data,i);');
    s = ones(1,nb_data)*nu(:,1);
    model.b(i) = (nu(:,1)'*model.ytrain(model.selector,i))./s;
    model.alpha(model.selector,i) = v(:,1)-(nu(:,1)*model.b(i));
end
return




