function plot_LS_boundary(x1, b, w)
% this should call after plot data points
% params: x1 x_axis range;
xmin = min(x1) * 1.20;
xmax = max(x1) * 1.20;
xx = linspace(xmin, xmax, 2);
yy = -(b + w(1)*xx)./w(2);
plot(xx,yy);
xlim([xmin,xmax]);
end
