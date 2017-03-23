clear; close all; 

%generate data  halfmoon(rad,width,d,n_samp)
dist = 5.0; 
width = 6; 
radius = 10; 
n_samp = 2000;
[data,data_shuffled] = halfmoon(radius, width,dist,n_samp);
X = data_shuffled(1:2,:);
class_train  = data_shuffled(3,:);

% one-hot encoding
y = zeros(10,size(class_train,2)); 
for i=1:size(class_train,2) 
	if class_train(i) == 1,  
		y(1,i) = 1.0;
	else 
		y(2,i) = 1.0;
	end 
end



%set up neural network archetecture 
% S = [5]; %number of neurons in hidden layer 
% % TF = {'tansig','tansig'}; %activation functions for each layer 
% TF = {'tansig','softmax'}; %activation functions for each layer 
% 
% %set net
% net = newff(X,y,S,TF);
% 
% %view net
% view(net); 
% %query about the net
% net.layers{2}.transferFcn 
% 
% 
% %set train parameters 
% net.trainParam.epochs = 100; 
% net.trainParam.goal = 1^-6;
% net.trainParam.mc = 0.0; 
% net.trainParam.max_fail = 1e8; 
% 
% %set validations 
% net.divideFcn = 'divideblock';
% net.divideParam.trainRatio = 0.70;
% net.divideParam.valRatio = 0.20; 
% net.divideParam.testRatio = 0.10; 
% 
% 
% %train network 
% [net1,tr1,Y1,E1,Pf1,Af1]=train(net,X,y);
% %test the trained net
% % predicts = sim(net1,X); 
% predicts = net1(X); 
% predict_y = zeros(size(predicts)); 
% predict_y( predicts>=0.5 ) = 1; 
% predict_y( predicts<0.5 ) = -1; 


% %get train accuracy rate 
% acc = sum(predict_y == y)/sum(size(y,2));
% fprintf('accuracy is %.3f \n', acc );


%pass 2, use feedforwardnet 
net = feedforwardnet(1, 'trainscg');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'softmax';
net.performFcn = 'crossentropy';
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;
[net,tr] = train(net,X,y);
predict2_y = net(X);



% %repeat using patternnet 
% net2 = patternnet(5); 
% net2 = train(net2,X,y ); 
% predict2_y = net2(X); 






