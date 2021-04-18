% Octave basics

B = randn(1,3); %This is going to have three values drawn % from a Gaussian Distribution with mean zero% and variance or Standard Deviation equal to one.

C = -6 + sqrt(10)*randn(1,6); 

% Vector with 6 element with mean = -6 and standard deviation = sqrt(10) 

hist(C)

A = [1 2; 3 4; 5 6]

size(A)

SZ = size(A)

% "SZ" is now a 1x2 vector(check with size(SZ)), containing [3, 2]

size(A, 1) % first dimension of matrix A.
size(A, 2) % second dimension of matrix A.

V = [1 2 3 4]

lenght(V) % gives the size of the longest dimension (in this case 4)

lenght(A) % = 3

lenght([1;2;3;4;5]) % = 5
