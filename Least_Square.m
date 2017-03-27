function [b,w]= Least_Square(X,y, lambda)
	%X in the form of [N,M]
    %y in the from of [N,1]

	[N,M] = size(X); 
	N_y = size(y); 

	if N~=N_y,
		error('X and y should have same samples!');
	end 


	XX = [ones(N,1),X]; 

	WW = inv(XX'*XX + lambda * eye(M+1)) * XX'*y; 

	b = WW(1); 
	w = WW(2:M+1);

end