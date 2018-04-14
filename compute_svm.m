function [w, bias, alpha, objective_function] = compute_svm(trD, trLb)
    run('vlfeat-0.9.21/toolbox/vl_setup.m');
    %[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();

    c = 10;

    x = trD;
    y = trLb;
    epsilon = 0.1;
    [d, n] = size(trD);
    f = ones(n, 1);
    f = -1 * f;
    h = zeros(n, n);

    for i = 1:n
        for j = 1:n
            h(i, j) = dot(x(:, i), x(:, j)) * y(i) * y(j);
        end
    end

    A = [];
    b = [];
    A_eq = trLb';
    b_eq = 0;
    lb = zeros(n, 1);
    ub = c * ones(n, 1);
    [alpha, f_val] = quadprog(h, f, A, b, A_eq, b_eq, lb, ub);

    %disp(f_val);
    temp = y .* alpha;
    w = x * temp;

    temp = abs(alpha - 0.05);
    [alpha_min, index] = min(temp);
    bias = y(index) - (w' * x(:, index));
    
    objective_function = -1 * f_val;