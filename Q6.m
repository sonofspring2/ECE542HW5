%%% using matlab neural network toolbox to solve the XOR problem
clear('all'); close all;


%%% generate data
X = [0,0;
     1,1;
     1,0;
     0,1]';
% t = [-1,-1,1,1];
t = [0,0,1,1];
% t = [0,1; 0,1; 1,0; 1,0]';

%%% set network parameters
% hidden_numbers = [config_hidden_number(config)];
hidden_numbers = 3;

%%% set net
net = patternnet(hidden_numbers);
%disable validation due to the limited train samples
net.divideFcn = '';
[net, tr]  = train(net,X,t);
y = net(X);

plotperform(tr);
s_title = sprintf('Q6 performance V.S. Epochs .jpg'); 
saveas(gcf, s_title);
