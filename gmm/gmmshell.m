function gmmshell
%% This is a shell for the GMM part of programming assignment 03.
 
%% first, load the data
  
X = load('hw3-data');

[N,D] = size(X);

%% for the dimensionality reduction part, run pca...you'll need to
%% preprocess the data a bit first

%X1 = ...preprocess X...
%Y = pca(X,2);

%% run GMM

K    = [2:10];   % values of k
CLL  = [];       % complete log likelihoods
CLLT = [];       % trace of complete log likelihoods for best run
C    = [];       % actual clusters
for k=K,
  CLL(k-1) = -Inf;
  CLLT(k-1,:) = zeros(1,10);
  C(k-1,:) = zeros(1,N);
  %% compute ten different clusterings with k components
  for iter=1:10,
    [c,cll,cllT] = gmm(X, k);
    
    %% record results
    if cll > CLL(k-1),
      CLL(k-1) = cll;
      CLLT(k-1,:) = cllT;
      C(k-1,:) = c;
      
      figure(2);
      plot(2:k,CLL(1:(k-1)),'b-');
 
      %% the following with plot the clusters for PCA:
      %figure(4);
      %plot(Y(c==1,1), Y(c==1,2), 'b.', ...
      %     Y(c==2,1), Y(c==2,2), 'r.', ...
      %     Y(c==3,1), Y(c==3,2), 'g.', ...
      %     Y(c==4,1), Y(c==4,2), 'k.', ...
      %     Y(c==5,1), Y(c==5,2), 'm.', ...
      %     Y(c==6,1), Y(c==6,2), 'c.', ...
      %     Y(c==7,1), Y(c==7,2), 'y.');
    end;
  end;
end;

%% plot results
figure(2);
plot(K,CLL,'b-');

figure(3);
for k=K,
  subplot(3,3,k-1);
  plot([1:10],CLLT(k-1,:),'b-');
end;



function [c,cll,cllT] = gmm(X,K)
  %% this should compute a clustering of X into K clusters.  the return
  %% values are: c = actual clustering ( c(n) = 1..K ), cll = complete
  %% log likelihood of final clustering, cllT = trace of cll over
  %% iterations
    
  [N,D] = size(X);
    
  %% initialize mu, Sigma and pi... you should use the "choose furthest
  %% data points heuristic for choosing mu
  pi = ...TODO...
  mu = zeros(K,D);
  Sigma = {};
  for k=1:K,
    mu(k,:) = ...TODO...
    Sigma{k} = eye(D);
  end;
  
  cllT = zeros(1,10);
  
  %% begin iterations
  for iter=1:10,
    %% expectation step: compute z

    %% here, sometimes we looe too much precision, so we compute
    %% *log* probabilities and then use the addLog function to
    %% sum them up
    
    z = zeros(N,K);
    for n=1:N,
      for k=1:K,
        z(n,k) = ...compute log probability...
      end;
      z(n,:) = exp(z(n,:) - addLog(z(n,:)));  % normalize
    end;
    
    %% maximization step
    
    %% first, re-estimate Sigma
    for k=1:K,
      Sigma{k} = ...TODO...
      
      %% we want to make sure Sigma doesn't get near singular
      %% if it does, just smooth it
      if cond(Sigma{k}) > 1e10,   % probably singular
        Sigma{k} = Sigma{k} + eye(D);
      end;
    end;
    
    %% second, re-estimate mu
    mu = ...TODO...
    
    %% finally, re-estimate pi
    pi = ...TODO...
    
    %% compute complete log-likelihood
    cllT(iter) = ...TODO...
    
    figure(1);
    plot(1:iter, cllT(1:iter));
  end;
  
  %% now, save the final cll
  cll = cllT(10);
  
  %% and save the final clustering
  c = zeros(1,N);
  for n=1:N,
    [x,k] = max(z(n,:));
    c(n) = k;
  end;
  

  
function l = log_normal(x,mu,si)
  l = ...TODO...

  %% you will want to use the "logdet" function below:


%% the remaining functions are for you to use as you please

function v = addLog(x)
  if length(x) == 0,
    v = -Inf;
  else
    [N,D] = size(x);
    v = log(sum(exp(x - repmat(max(x),N,1)))) + max(x);
  end;

function y = logdet(A)
% log(det(A)) where A is positive-definite.
% This is faster and more stable than using log(det(A)).
  U = chol(A);
  y = 2*sum(log(diag(U)));
