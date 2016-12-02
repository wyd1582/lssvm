function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars)

% Construct the positive (semi-) definite and symmetric kernel matrix
%
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
%
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
%
%
%
% Full syntax
%
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
%
% Outputs
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs
%   X      : N x d matrix with the inputs of the training data
%   kernel : Kernel type (by default 'RBF_kernel')
%   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%
% See also:
%  RBF_kernel, lin_kernel, kpca, trainlssvm, kentropy

[nb_data,d] = size(Xtrain);

if strcmp(kernel_type,'RBF_kernel'),
    
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(2*kernel_pars(1)));
   
    
elseif strcmp(kernel_type,'RBF4_kernel'),
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
    
elseif strcmp(kernel_type,'lin_kernel')
        omega = Xtrain*Xtrain';
   
elseif strcmp(kernel_type,'poly_kernel')
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
end