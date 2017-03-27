%%% kernel non-linear transformation for a simple 1D case
clear; close all;


x = [0;2;1;3];
y = [1;1;-1;-1];

%non-linear tranform to 2D
x_cs  = [0,2];
sigmas = [1,1];

X = zeros(length(x), 2);
for i = 1:size(X,1)
  for j=1:size(X,2)
    X(i,j) = radial_function(x(i),x_cs(j), 1);
  end
end


[b,w] = Least_Square(X,y,0);


%determine boundary point in the original points
xs = linspace(0,3,100);
for i=2:length(xs)
  Xs = zeros(2,2);
  for j = 1:2
    Xs(1,j) = radial_function(xs(i-1),x_cs(j), 1);
    Xs(2,j) = radial_function(xs(i),x_cs(j), 1);
  end
  pred1 = w(1)*Xs(1,1) +w(2)*Xs(1,2) + b;
  pred2 = w(1)*Xs(2,1) +w(2)*Xs(2,2) + b;

  if pred1* pred2 < 0
    fprintf('boundary point: %.3f \n',xs(i));
  end
end



%plot transformed result
figure(1);
box on;
hold on;
X_1 = X(y==1,:);
scatter(X_1(:,1), X_1(:,2), 'r');
X_1 = X(y==-1,:);
scatter(X_1(:,1), X_1(:,2), 'b');
xlabel('\phi_1');
ylabel('\phi_2');
legend('label=1','label=-1');

plot_LS_boundary([-1,2],b,w);
saveas(gcf, 'Q4.jpg');
