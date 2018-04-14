%{
run('vlfeat-0.9.21/toolbox/vl_setup.m');
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();

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
%}

load("trainAnno.mat");
c = 10;
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
[w, bias, alpha, objective_function] = compute_svm(trD, trLb);
[d, n] = size(trD);
PosD = [];
NegD = [];
epsilon = 0.1;

for i = 1:size(trD, 2)
   if trLb(i) == 1
       PosD = [PosD, trD(:, i)];
   else
       if alpha(i) < epsilon
           NegD = [NegD, trD(:, i)];
       end
   end
end

objective_function_vals = [];
ap_array = [];

for iter = 1:10
    disp("iteration : ");
    disp(iter);
    PosD = [];
    NegD = [];
    for i = 1:size(trLb, 1)
       if trLb(i) == 1
           PosD = [PosD, trD(:, i)];
       else
           if alpha(i) < epsilon
               NegD = [NegD, trD(:, i)];
           end
       end
    end
    HW2_Utils.genRsltFile(w, bias, "train", "question_4_4_2_rects");
    load("question_4_4_2_rects.mat");
    hard_neg = [];
    for i = 1:length(rects)
        im = imread(sprintf('%s/%sIms/%04d.jpg', HW2_Utils.dataDir, "train", i));
        [imH, imW,~] = size(im);
        current_rect = rects{i};
        badIdxs = or(current_rect(3,:) > imW, current_rect(4,:) > imH);
        current_rect = current_rect(:,~badIdxs);
        ubs = ubAnno{i};
        overlaps = [];
        for j = 1:size(ubs, 2)
            ov_rect = HW2_Utils.rectOverlap(current_rect, ubs(:, j));
            overlaps = [overlaps, ov_rect];
        end        
        
        for j = 1:length(current_rect)
            if current_rect(5, j) > 0
               continue 
            end
            break_flag = 0;
            for k = 1:size(ubs, 2)
                if overlaps(j, k) > 0.3
                    break_flag = 1;
                    break;
                end
            end
            if break_flag == 0
              
                imReg = im(int16(current_rect(2, j)):int16(current_rect(4, j)), int16(current_rect(1, j)):int16(current_rect(3, j)), :);
                imReg = imresize(imReg, HW2_Utils.normImSz);
                
                feat = HW2_Utils.cmpFeat(rgb2gray(imReg));
                feat = feat / norm(feat);
                hard_neg = [hard_neg, feat];
                
                if size(hard_neg, 2) > 1000
                    break;
                end
            end
            if size(hard_neg, 2) > 1000
                break;
            end
        end
        if size(hard_neg, 2) > 1000
            break;
        end
    end
    NegD = [NegD, hard_neg];
    temp_neg_labels = -1 * ones(size(NegD, 2), 1);
    trD = [];
    trD = [trD, PosD];
    trLb = ones(size(trD, 2), 1);
    trD = [trD, NegD];
    trLb = [trLb; temp_neg_labels];
    
    %disp(size(trD));
    %disp(size(trLb));
    [w, bias, alpha, objective_function] = compute_svm(trD, trLb);
    %disp(size(alpha));
    %{
    objective_function_part_1 = (norm(w) ^ 2) / 2;
    summation = 0;
    for j = 1:size(trLb, 1)
        summation = summation + max((1 - trLb(j) * (w' * trD(:, j) + bias)), 0);
    end
    %}
    objective_function_vals = [objective_function_vals, objective_function];
    
    HW2_Utils.genRsltFile(w, bias, "val", "question_4_4_2_val_outputs");

    [ap, prec, rec] = HW2_Utils.cmpAP("question_4_4_2_val_outputs", "val");
    ap_array = [ap_array, ap];
end
numbers = linspace(1, 10, 10);
subplot(2,1,1);
plot(numbers, objective_function_vals);
subplot(2,1, 2);
plot(numbers, ap_array);

HW2_Utils.genRsltFile(w, bias, "test", "submission_output_test");


