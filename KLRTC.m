function [T_recon] = KLRTC(T, Xi, R, Mask, Max_Iter, Alpha,kernel,d,Sigma)
%Inputs:
%     'T' - Tensor type. The incomplete tensor that should be completed.
%            You need to install the tensor toolbox: http://www.tensortoolbox.org/
%
%     'Xi' - Cell type. The factor matrices for the decomposition. 
%         For example:
%            Xi = cell(2,n_dim);
%            for k = 1:n_dim
%               % Initialize factor matrices
%               Xi{1,k} = rand(dim(k), R{k});
%               Xi{2,k} = rand(R{k}, prod(dim)/dim(k));
%            end
%  
%     'R' - Cell type. The num of latent features for each mode, 
%           e.g., R={10,10,10}
%
%     'Mask' - Tensor type. A binary tensor that has the same dimension as T
%     
%     'Max_Iter' - Maximum iteration steps, integer
%
%     'Alpha' - Cell type. The regularization parameters for each mode,
%            e.g., Alpha = {100,100,100}
%
%     'kernel' - String type, specify which kernel funtion to use,
%              e.g., "Linear", "Poly", and "RBF"
%              You can add your own
%
%     'd' - Integer (>=2), parameter for polynomial kernel
%     'Sigma' - Positive real number, parameter for RBF kernel
%
%Output:
%     T_recon: the completed tensor
%
% Liping Deng
% liping.deng@siu.edu
% 9/24/2022
% Please see the paper for the algorithm details
%%
dim = size(T);
n_dim = length(dim);
Ti = cell(1,n_dim);
T_recon = T;

for j = 1:Max_Iter
    T_recon = Mask.*T+(1-Mask).*T_recon;
    for k = 1:n_dim
        % Matrize tensor along each mode
        Ti{k} = double(tenmat(T_recon,k));
    end
    
    T_recon = tensor(zeros(size(T)));
    for i = 1:n_dim
        II = Alpha{i}*eye(R{i});
        if strcmp(kernel, 'Linear') == 1
            Xi{1,i} = (Ti{i}*Xi{2,i}')/(Xi{2,i}*Xi{2,i}'+II);
            Xi{2,i} = (Xi{1,i}'*Xi{1,i}+II)\(Xi{1,i}'*Ti{i});
            Ti{i} = Xi{1,i}*Xi{2,i};
        elseif strcmp(kernel, 'Poly') == 1
            [Kww, Kxw, Kxx_p, Kww_p, Kxw_p] = kernel_matrix(Ti{i}, Xi{1,i}, kernel, d, Sigma);
            Xi{2,i} = (Kww+II)\Kxw';     
            HessenW = (Xi{2,i}*Xi{2,i}').*Kww_p+II;
            C = Kxw_p.*Xi{2,i}';
            Xi{1,i} = Ti{i}*C/HessenW;
            
            HessenX = Kxx_p.*eye(size(Ti{i},2));
            Ti{i} = Xi{1,i}*C'/(HessenX);
        elseif strcmp(kernel, 'RBF') == 1
            [Kww, Kxw, ~, ~, ~] = kernel_matrix(Ti{i}, Xi{1,i}, kernel, d, Sigma);
            Xi{2,i} = (Kww+II)\Kxw';
            
            Q1 = Xi{2,i}'.*Kxw; Q2 = (Xi{2,i}*Xi{2,i}').*Kww;
            Tao1 = diag(ones(1,size(Ti{i},2))*Q1); Tao2 = diag(ones(1,R{i})*Q2);
            HessenW = Q2+Tao1-Tao2+(Sigma/2)*II;
            Xi{1,i} = Ti{i}*Q1/HessenW;
            
            Q3 = Q1';
            Tao3 = diag(ones(1,R{i})*Q3);
            HessenX = Tao3;
            Ti{i} = Xi{1,i}*Q3/HessenX;
        end
        T_recon = T_recon+tensor(tenmat(Ti{i},i,[1:i-1,i+1:n_dim],dim))*(1/n_dim);
    end
    err = norm(T-T_recon)/norm(T);
    if err<=Tol
        break;
    end
end
% % T_recon = T_recon.*(1-Mask)+T.*Mask;
end

function [Kww, Kxw, Kxx_p, Kww_p, Kxw_p] = kernel_matrix(X, W, kernel, d, gamma)
c = 1;
if strcmp(kernel,'Poly')==1
    Kww = (W'*W+c).^d;
    Kxw = (X'*W+c).^d;
    Kxx_p = (X'*X+c).^(d-1);
    Kww_p = (W'*W+c).^(d-1);
    Kxw_p = (X'*W+c).^(d-1);
    %     Kxx = [];
elseif strcmp(kernel,'RBF')==1
    Kxw = exp(-(dist(X',W).^2)/gamma);
    Kww = exp(-(dist(W',W).^2)/gamma);
    %     Kxx = exp(-(dist(X',X).^2)/gamma);
    %     Kxx = [];
    Kww_p = [];
    Kxw_p = [];
    Kxx_p = [];
end
end