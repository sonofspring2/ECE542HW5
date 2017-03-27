%%% kernel non-linear transformation for a simple 1D case
clear; close all;


x = [0;2;1;3];
y = [1;1;-1;-1];

%non-linear tranform to 2D
x_cs  = [0,2,1];
sigmas = [1,1,100];

X = zeros(length(x), length(x_cs));
for i = 1:size(X,1)
  for j=1:size(X,2)
    X(i,j) = radial_function(x(i),x_cs(j), sigmas(j));
  end
end

w = [1,1,-1]';
r = X *w;

%determine boundary point in the original points
xs = linspace(0,3,100);
for i=2:length(xs)
  Xs = zeros(2,3);
  for j = 1:3
    Xs(1,j) = radial_function(xs(i-1),x_cs(j), sigmas(j));
    Xs(2,j) = radial_function(xs(i),x_cs(j), sigmas(j));
  end
  % pred1 = w(1)*Xs(1,1) +w(2)*Xs(1,2) + b;
  % pred2 = w(1)*Xs(2,1) +w(2)*Xs(2,2) + b;
  preds = Xs *w;
  pred1 = preds(1);
  pred2 = preds(2);

  if pred1* pred2 < 0
    fprintf('%.3f \n',xs(i));
  end
end
