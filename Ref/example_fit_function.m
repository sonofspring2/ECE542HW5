% neural network example for curve fit
% uses validation 
% net = newff(P,T,[S1 S2...S(N-l)],{TF1 TF2...TFNl},...
% BTF,BLF,PF,IPF,OPF,DDF)
%
% P
% 	R x Q1 matrix of Q1 sample R-element input vectors
% T
% 	SN x Q2 matrix of Q2 sample SN-element target vectors
% Si
% 	Size of ith layer, for N-1 layers, default = [ ].
% (Output layer size SN is determined from T.)
% TFi
% 	Transfer function of ith layer. (Default = 'tansig' for
% hidden layers and 'purelin' for output layer.)
% BTF
% 	Backpropagation network training function (default = 'trainlm')
% BLF
% 	Backpropagation weight/bias learning function (default = 'learngdm')
% PF
% 	Performance function. (Default = 'mse')
% IPF
% 	Row cell array of input processing functions. (Default = {'fixunknowns','removeconstantrows','mapminmax'})
% OPF
% 	Row cell array of output processing functions. (Default = {'removeconstantrows','mapminmax'})
% DDF
% 	Data divison function (default = 'dividerand')



%       P = [0 1 2 3 4 5 6 7 8 9 10];
%       T = [0 1 2 3 4 3 2 1 2 3 4];

% example of functional approximation
dt = 0.001;  % sampling interval
t = 0:dt:1;  % input times
N_samples = size(t,2);  % number of total samples
percent_train = 5; % percent samples of total to use for training
N_train = fix(N_samples*percent_train/100); % number of training samples
f = 2*cos(2*pi*3*t);  % complete function to be fitted
S = [5];  % number of neurons in hidden layer
TF = {'tansig','purelin'}; % activation functions for each layer
	% second function is for output layer
dx = fix(N_samples/N_train);
ind_train = 1:dx:N_samples;
tp = t(ind_train);  % training input
fp = f(ind_train);  % training output (desired)
net = newff(tp,fp,S,TF); % call to set up network 
 
net.trainParam.epochs = 100; % set max number of epochs
net.trainParam.goal = 10^-6;
% net.trainParam.lr = 0.01; %learning rate
net.trainParam.mc = 0.0; %momentum parameter
net.trainParam.max_fail = 10^8;  % number of validation failures

% net.divideFcn = 'dividetrain';  % no validation samples
net.divideFcn = 'divideblock';  % blockwise division of training/validation samples
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.20;
net.divideParam.testRatio = 0.10;
[net1,tr1,Y1,E1,Pf1,Af1] = train(net,tp,fp);  % train the network	subsample	% , default method is Levenberg-Marquardt
y1 = sim(net1,t); % run the network with input t, testing set



% use permuted input
N_samples = size(t,2);
rnd = rand(N_samples);
[rnd_sort,ind] = sort(rnd);
% tp = permute(t,ind);
% fp = permute(f,ind);
ind_train = ind(1:N_train);
tp = t(ind_train);  % training input
fp = f(ind_train);  % training output (desired)
net = newff(tp,fp,S,TF);
net.trainParam.epochs = 100;  % train random
net.trainParam.goal = 10^-6;
net.trainParam.max_fail = 10^8;  % number of validation failures
% net.divideFcn = 'dividetrain';  % no validation samples
net.divideFcn = 'divideblock';  % blockwise division of training/validation samples
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.20;
net.divideParam.testRatio = 0.10;
[net2,tr2,Y2,E2,Pf2,Af2] = train(net,tp,fp);
y2 = sim(net2,t); % run the network with input t, testing set
figure(1)
plot(t,y1,'b',t,y2,'r',t,f,'--')
title('compare in time domain')
legend('unifromly sampled','random sampled', 'orignal');

Nepoch1 = size(E1,2);
Nepoch2 = size(E2,2);
ep1 = 1:Nepoch1;
ep2 = 1:Nepoch2;
figure(2)
n1=1:size(tr1.perf,2); % tr1.perf contains the mse at each epoch
n2=1:size(tr2.perf,2); % tr2.perf contains the mse at each epoch
semilogy(n1,tr1.perf,'b',n2,tr2.perf,'r')
title('perf(MSE) as function of epoch')
legend('unifromly sampled','random sampled');

mse1 = mean(E1.^2);
mse2 = mean(E2.^2);
fprintf('mse1 = %f,   mse2 = %f \n', mse1,mse2);




