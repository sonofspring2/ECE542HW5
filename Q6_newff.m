%%% using matlab neural network toolbox to solve the XOR problem
clear('all'); close all;


%%% generate data
X = [0,0;
     1,1;
     1,0;
     0,1]';
t = [-1,-1,1,1];
% t = [0,0,1,1];
% t = [0,1; 0,1; 1,0; 1,0]';

%%% set network parameters
% hidden_numbers = [config_hidden_number(config)];
% hidden_numbers = 3;

% %%% set net
% net = patternnet(hidden_numbers);
% %disable validation due to the limited train samples
% net.divideFcn = '';
% [net, tr]  = train(net,X,t);
% y = net(X);
%%% set network parameters

train_t = X;
train_f = t;
test_t = X;

hidden_numbers = [4];
act_functions = {'tansig', 'purelin'};

%%% set up network
net = newff(train_t, train_f, hidden_numbers, act_functions);

%%% set trainFun
%     net.trainFcn = 'traincgf'; % gradient decent

%%% set training parameters
net.trainParam.epochs = 20000; % set max number of epochs
net.trainParam.goal = 10^-20;

%     net.trainParam.lr = 0.01; %learning rate
net.trainParam.lr = 0.1; %learning rate
net.trainParam.mc = 0.0; %momentum parameter
net.trainParam.max_fail = 10^8;  % number of validation failures

% net.divideFcn = 'dividetrain';  % no validation samples
net.divideFcn = '';  % blockwise division of training/validation samples


%%% train the network and predict
[net1, tr1, Y1, E1, Pf1, Af1] = train(net,train_t, train_f);
predict_f  = sim(net1,test_t);



plotperform(tr1);
s_title = sprintf('Q6_performanceV.S.Epochs.jpg');
saveas(gcf, s_title);


%%% plot boundary;
x_min = -1; x_max = 2; 
y_min = -1; y_max = 2;
[x_b, y_b] = meshgrid(x_min:(x_max-x_min)/30:x_max,y_min:(y_max-y_min)/30:y_max);

z_b =zeros(size(x_b));
% predict
for i=1:size(x_b,1)
  for j=1:size(x_b,2)
    z_b(i,j) = sim(net1,[x_b(i,j);y_b(i,j)]);
  end
end

figure;
%Adding colormap to the final figure
sp = pcolor(x_b,y_b,z_b);
load red_black_colmap;
colormap(red_black);
shading flat;

box on; 
hold on; 
scatter([0,1],[0,1],'w');
scatter([1,0],[0,1],'b');
contour(x_b,y_b,z_b,[0 0],'k','Linewidth',1);
s_title = sprintf('Q6_boundary.jpg');
saveas(gcf, s_title);

