clear; close all;

%generate data  halfmoon(rad,width,d,n_samp)
dist = 5.0;
width = 6;
radius = 10;
n_samp = 2000;
[data,data_shuffled] = halfmoon(radius, width,dist,n_samp);
X = data_shuffled(1:2,:);
class_train  = data_shuffled(3,:);
y = class_train;

[data,data_shuffled] = halfmoon(radius, width,dist,n_samp);
test_x = data_shuffled(1:2,:);
test_y = data_shuffled(3,:);


% set up repeat run parameters
repeat = 5;
hidden_numbers = [5,20];
trainFcns = {'trainlm', 'traingd', 'traincgf'};



for j=1:2
  for k=1:3
    print_config = true;
    for i=1:repeat
      %set up neural network archetecture
      S = [hidden_numbers(j)]; %number of neurons in hidden layer
      TF = {'tansig','purelin'}; %activation functions for each layer

      %set net
      net = newff(X,y,S,TF);
      %set trainFcn
      net.trainFcn = trainFcns{k};

      %set train parameters
      net.trainParam.epochs = 10000;
      net.trainParam.goal = 1e-6;
      net.trainParam.mc = 0.0;
      net.trainParam.max_fail = 1e8;
      net.trainParam.lr = 0.1; %learning rate


      %set validations
      net.divideFcn = 'divideblock';
      net.divideParam.trainRatio = 0.80;
      net.divideParam.valRatio = 0.10;
      net.divideParam.testRatio = 0.10;

      start_t = cputime;
      %train network
      [net1,tr1,Y1,E1,Pf1,Af1]=train(net,X,y);
      end_t = cputime;
      % fprintf('Using time is %.3fs \n', end_t - start_t);
      predicts = net1(test_x);

      %%% print out statistics
      if print_config
        fprintf('parameters: hidden_numbers = %d, trainFcn = %s\n', S(1), net.trainFcn);
        print_config = false;
      end

      N_epoch = size(tr1.perf,2) -1;
      res = predicts - test_y;
      mse = mean(res.^2);
      fprintf('MSE = %.3e, Epochs = %d, cputime= %.3fs\n', mse, N_epoch, end_t - start_t);
    end
  end
end
