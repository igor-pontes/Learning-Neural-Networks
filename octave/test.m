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

% load('data1.dat') - loads file with your data and creates, in this case, a var named
% "data1", with your data.

who % variables in the current scope.
whos % show variables in the current scope with detailed information

V(1:3) % displays the first 3 items of the vector "V"

% save test.mat V; - will save variable "V" in a file called test.mat in binary format.

clear - % deletes all the variables in the workspace;

% save hello.txt v -ascii - saves data in a more human readable format (ASCII).
A = [1 2; 3 4; 5 6]

A(3, 2) % gives us the A3,2 element. (= 6)

% A(2,:) - ":" means every element along that row/column

% A(:,2) - gives us everything in the second column

A([1 3], :) % gives us every element in the first and third row

A(:,2) = [10;11;12] % taking the second column of "A" and assigning this vector to it

A = [A, [100; 101; 102]]; % append another column vector to the right of "A"

% [100; 101; 102] - column vector

A(:) % put all elements of "A" into a single column vector

A = [1 2; 3 4; 5 6];

B = [11 12; 13 14; 15 16];

C = [A B] % concatenate matrix "A" and "B" to eachother

C = [A; B] % instead of concatenating them horizontally, concatenate them vertically 

% [A B] == [A, B] - same result
