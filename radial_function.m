function out = radial_function( x, x_c, sigma )
% function: Short description
% radial function for non-linear transformation
% Extended description

% sigma = 1;
dis = x - x_c;

out = exp(-dis' * dis/ sigma );

end  % function
