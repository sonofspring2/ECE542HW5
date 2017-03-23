%problem 3
% fit function using matlab neural network toolbox

clear; close all;

%%%set train and test data
dt = 0.001;  % sampling interval
t = 0:dt:1;  % input times
N_samples = size(t,2);  % number of total samples
f = sin(2*pi*2.5*t + pi/4);


train_ratio = 0.8;
test_ratio = 0.2;
[trainIdx,valIdx, testIdx] = dividerand(N_samples,train_ratio,0,test_ratio);

%train data
rnd  = randperm(size(trainIdx,2) );
train_t_ = t(trainIdx);
train_t = train_t_( rnd );
train_f_ = f(trainIdx);
train_f = train_f_( rnd );


%test data
rnd  = randperm(size(testIdx,2) );
test_t_  = t(testIdx);
test_t = test_t_(rnd);
test_f_  = f(testIdx);
test_f = test_f_(rnd);


%%% set network parameters
hidden_numbers = [5];
act_functions = {'tansig', 'purelin'};

%%% set up network
net = newff(train_t, train_f, hidden_numbers, act_functions);


%%% set training parameters
net.trainParam.epochs = 100; % set max number of epochs
net.trainParam.goal = 10^-6;
net.trainParam.lr = 0.01; %learning rate
net.trainParam.mc = 0.0; %momentum parameter
net.trainParam.max_fail = 10^8;  % number of validation failures

% net.divideFcn = 'dividetrain';  % no validation samples
net.divideFcn = 'divideblock';  % blockwise division of training/validation samples
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.20;
net.divideParam.testRatio = 0.10;

%%% train the network and predict
[net1, tr1, Y1, E1, Pf1, Af1] = train(net,train_t, train_f);
% predict_f  = sim(net1,test_t);


