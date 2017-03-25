%problem 3
% fit function using matlab neural network toolbox

clear; close all;

%%% set repeat parameters
Repeat = 1;
Config_number = 2;
config_hidden_number = [5,20];

for config = 1:Config_number,
  plot_param = true;
  for i=1:Repeat,

    %%%set train and test data
    dt = 0.001;  % sampling interval
    t = 0:dt:1;  % input times
    N_samples = size(t,2);  % number of total samples
    f = sin(2*pi*2.5*t + pi/4);

    N_train = 100;
    train_ratio = N_train/size(t,2);
    test_ratio = 1- train_ratio;
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
    hidden_numbers = [config_hidden_number(config)];
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
    predict_f_  = sim(net1,test_t_);


    %%%plot result
    figure(1);
    box on;
    scatter(train_t,train_f,30,'k');
    hold on;
    plot(t,f,'r','Linewidth',1.5);
    plot(test_t_,predict_f_,'b','Linewidth',1.5);
    legend('train sample ','original data', 'predict result');
    s_title = sprintf('Sin function fit result. trainRatio = 0.8, MSE = %.3e ', mean(E1.^2));
    title(s_title);
    xlabel('x');
    ylabel('y');
    s_filename = sprintf('Q3_hidden_number_%d, %d.png',hidden_numbers,i);
    saveas(gcf,s_filename);

    %plot learning curve
    figure(2);
    box on ;
    n1 = 1:size(tr1.perf, 2);
    semilogy(n1,tr1.perf,'r');
    xlabel('epoch number');
    ylabel('MSE');
    s_title = sprintf('perf(MSE) as function of epoch.hiddenNumbers = %d ', hidden_numbers);
    title(s_title);
    s_filename = sprintf('Q3_hidden_number_learing_curve_%d.png', hidden_numbers,i);
    saveas(gcf,s_filename);


    %%% print out statistics
    if plot_param
      fprintf('parameters: hidden_numbers = %d \n', hidden_numbers);
      plot_param = false;
    end
    %use tr1.perf to infer epoch number
    N_epoch = size(tr1.perf,2) - 1 ;
    fprintf('MSE = %.3e, Epochs = %d \n', mean(E1.^2), N_epoch);

  end
end
