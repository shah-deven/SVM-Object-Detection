
run('vlfeat-0.9.21/toolbox/vl_setup.m');
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();

c = 10;

x = trD;
y = trLb;

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

y_pred = (w' * valD) + bias;
[vn, dummy] = size(y_pred);
for i = 1:vn
    if y_pred(i) < 0
        y_pred(i) = -1;
    else
        y_pred(i) = 1;
    end
end

correct_predictions = 0;
wrong_predictions = 0;

for i = 1:vn
    if y_pred(i) == valLb(i)
        correct_predictions = correct_predictions + 1;
    else
        wrong_predictions = wrong_predictions + 1;
    end
end
disp(correct_predictions);
disp(wrong_predictions);

HW2_Utils.genRsltFile(w, bias, "val", "question_4_4_1_output")

[ap, prec, rec] = HW2_Utils.cmpAP("question_4_4_1_output", "val");

