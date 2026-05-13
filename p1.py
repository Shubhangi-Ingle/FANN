program 13
disp('Enter weights');
w1 = input('Weight w1 = ');
w2 = input('Weight w2 = ');

disp('Enter Threshold Value');
theta = input('theta = ');

y = [0 0 0 0];
x1 = [0 0 1 1];
x2 = [0 1 0 1];

% Desired output for ANDNOT function
z = [0 0 1 0];

con = 1;

while con == 1
    
    % Calculate net input
    zin = x1*w1 + x2*w2;
    
    for i = 1:4
        
        if zin(i) >= theta
            y(i) = 1;
        else
            y(i) = 0;
        end
        
    end

    disp('Output of Net');
    disp(y);

    % Check whether output matches desired output
    if isequal(y, z)
        con = 0;
    else
        disp('Net is not learning. Enter another set of weights and Threshold value');

        w1 = input('Weight w1 = ');
        w2 = input('Weight w2 = ');
        theta = input('theta = ');
    end

end

disp('McCulloch-Pitts Net for ANDNOT function');

disp('Weights of Neuron');
disp(w1);
disp(w2);

disp('Threshold value');
disp(theta);














program 12

x = -10:0.1:10;

% Logistic Function
tmp = exp(-x);
y1 = 1 ./ (1 + tmp);

% Hyperbolic Tangent Function
y2 = (1 - tmp) ./ (1 + tmp);

% Identity Function
y3 = x;

% Logistic Function Plot
subplot(2,3,1);
plot(x, y1);
grid on;
axis([min(x) max(x) -2 2]);
title('Logistic Function');
xlabel('(a)');
axis square;

% Hyperbolic Tangent Function Plot
subplot(2,3,2);
plot(x, y2);
grid on;
axis([min(x) max(x) -2 2]);
title('Hyperbolic Tangent Function');
xlabel('(b)');
axis square;

% Identity Function Plot
subplot(2,3,3);
plot(x, y3);
grid on;
axis([min(x) max(x) min(x) max(x)]);
title('Identity Function');
xlabel('(c)');
axis square;
