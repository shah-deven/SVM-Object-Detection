classdef HW2_Utils
% Helper functions to load, display data, and compute feature vectors
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 11-Feb-2016
% Last modified: 11-Feb-2016

    properties (Constant)        
        dataDir = '../hw2data';
        
        % Anotated upper bodies have different sizes. To train a classifier, we need to normalize to
        % a standard size
        normImSz = [64, 64];
        
        % The size of HOG cell for HOG feature computation. 
        % This must be divisible by normImSz(1) and normImSz(2)
        hogCellSz = 8;
    end
    
    methods (Static)
        % Show some random images and upper body annotation
        function demo1()
            fprintf('Display random images with upper body annotation\n');
            load(sprintf('%s/trainAnno.mat', HW2_Utils.dataDir), 'ubAnno');            
            nR = 3; nC = 4;
            idxs = randsample(length(ubAnno), nR*nC);            
            for i=1:nR*nC
                idx = idxs(i);
                im = sprintf('%s/trainIms/%04d.jpg', HW2_Utils.dataDir, idx);
                subplot(nR, nC, i); imshow(im);
                if ~isempty(ubAnno{idx})
                    HW2_Utils.drawRects(ubAnno{idx});
                end;                    
            end
        end;
        
        % Display HOG features for random positive and negative images
        function demo2()
            fprintf('Display a random training images and correspond HOG features\n');            
            [trD, trLb, trRegs]   = HW2_Utils.getPosAndRandomNegHelper('train');
            
            nR = 4; nC = 4;
            idxs = randsample(length(trLb), nR*nC);            
            
            trD = single(trD);
            nCellPerSide = round(sqrt(size(trD,1)/31)); % assume HOG dim is nCellPerSide*nCellPerSide*31
            
            for i=1:(nR*nC)/2
                idx = idxs(i);
                im = trRegs(:,:,:,idx);
                subplot(nR, nC, 2*i-1); imshow(im);
                if trLb(idx) == 1
                    strTitle = 'Positive - Upper body';
                else
                    strTitle = 'Negative - Not Upper body';
                end
                title(strTitle);
                
                hogFeat = trD(:,idx);                
                hogFeat = reshape(hogFeat, [nCellPerSide, nCellPerSide, 31]);
                hogIm = vl_hog('render', hogFeat);
                subplot(nR, nC, 2*i); imshow(hogIm);
                title(strTitle);
            end            
        end;
        
        % Get the train and validation data and annotation to train the classifier
        % Positive instances: all annotated upper bodies
        % Negative instances: random image patches
        function [trD, trLb, valD, valLb, trRegs, valRegs] = getPosAndRandomNeg()
            cacheFile = sprintf('%s/trainval_random.mat', HW2_Utils.dataDir);
            if exist(cacheFile, 'file')
                load(cacheFile);
            else
                [trD, trLb, trRegs]   = HW2_Utils.getPosAndRandomNegHelper('train');
                [valD, valLb, valRegs] = HW2_Utils.getPosAndRandomNegHelper('val');
                save(cacheFile, 'trD', 'trLb', 'valD', 'valLb', 'trRegs', 'valRegs');
            end;
            trD = HW2_Utils.l2Norm(double(trD));
            valD = HW2_Utils.l2Norm(double(valD));
        end

        % Perform sliding window detection and return a list of rectangular regions with scores
        % w, b: weight and bias term of SVM
        % rects: 5*k matrix for k detections, sorted by detection scores
        %   rects(:,i) is left, top, right, bottom, detection score
        function rects = detect(im, w, b, shldDisplay)
            winSz = HW2_Utils.normImSz;
            
            smallestUbSize = 45; % desired smallest upper body size we can detect
            biggestUbSize = 310; % desired biggest  upper body size we can detect

            % To detect upper bodies of multiple sizes, we need to run the detector on the resized
            % image at multiple scales. Here we determine the scales to run detection. 
            % We use log-scale, ie, we increase the size of the image by a constant ratio from one
            % scale to the next.             
            smallestScale = winSz(1)/biggestUbSize;
            biggestScale = winSz(1)/smallestUbSize;                        
            scaleStep = log(1.2); 
            scales = exp(log(smallestScale):scaleStep:log(biggestScale));
            
            grayIm = rgb2gray(im); % convert to rgb
            
            rects = cell(1, length(scales)); % store the 
            for s = 1:length(scales) % consider each scale in turn
                % Resize the image to a particular scale
                scale = scales(s);
                scaleIm = imresize(grayIm, scale);
                
                % Compute the HOG image
                hogIm = vl_hog(im2single(scaleIm), HW2_Utils.hogCellSz);
                
                % Consider multiple subwindows of size winSz4HogIm with step size 1 (i.e, sliding window)
                % ML_SlideWin is an efficient way to do sliding window. It considers multiple 
                % subwindows at the same time, avoiding for loop. 
                winSz4HogIm = [winSz/HW2_Utils.hogCellSz, size(hogIm,3)];
                stepSz = [1 1 1];
                obj = ML_SlideWin(hogIm, winSz4HogIm, stepSz);
                
                nBatch = obj.getNBatch;
                [topLefts_s, scores_s] = deal(cell(1, nBatch));
                for i=1:nBatch
                    [hogFeats, topLefts_s{i}] = obj.getBatch(i);
                    D = HW2_Utils.l2Norm(hogFeats);
                    scores_s{i} = D'*w + b;
                end;
                scores_s   = cat(1, scores_s{:});
                topLefts_s = cat(2, topLefts_s{:});             
                
                % From HOG image coordinate to scaled image coordinate
                topLefts_s = (topLefts_s([2,1],:)-1)*HW2_Utils.hogCellSz + 1;                                
                rects_s = [topLefts_s; topLefts_s + repmat(winSz' -1, 1, size(topLefts_s,2))];
                                
                % convert back to coordinate of scale image                
                rects_s = rects_s/scale;
                                
                % Append score
                rects{s} = [rects_s; scores_s'];
            end
            rects = cat(2, rects{:});   
            
            % Perform non-maxima suppression, i.e., remove detections that have high overlap (0.5)
            % with another detection with a higher score            
            rects = HW2_Utils.nms(rects, 0.5); 
            
            % Display the top 4 detections if asked
            if exist('shldDisplay', 'var') && shldDisplay                
                imshow(im);            
                HW2_Utils.drawRects(rects(:,1:4));
            end;
        end;
        
        % Generate detection result file for a particular dataset
        % dataset: either 'train', 'val', or 'test'
        % outFile: path to save the result.
        function genRsltFile(w, b, dataset, outFile)
            imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', HW2_Utils.dataDir, dataset), 'jpg');
            nIm = length(imFiles);            
            rects = cell(1, nIm);
            startT = tic;
            for i=1:nIm
                ml_progressBar(i, nIm, 'Ub detection', startT);
                im = imread(imFiles{i});
                rects{i} = HW2_Utils.detect(im, w, b);                                
            end
            save(outFile, 'rects');
            fprintf('results have been saved to %s\n', outFile);
        end;


        
        % Calculate the Average precision for a given result file and dataset
        % This requires the annotation file is available for the dataset. 
        function [ap, prec, rec] = cmpAP(rsltFile, dataset)
            load(sprintf('%s/%sAnno.mat', HW2_Utils.dataDir, dataset), 'ubAnno');
            load(rsltFile, 'rects');
            
            if length(rects) ~= length(ubAnno)
                error('result and annotation files mismatch. Are you using the right dataset?');
            end
            
            nIm = length(ubAnno);
            [detScores, isTruePos] = deal(cell(1, nIm));            
            
            for i=1:nIm
                rects_i = rects{i};
                detScores{i} = rects_i(5,:);
                ubs_i = ubAnno{i}; % annotated upper body
                isTruePos_i = -ones(1, size(rects_i, 2));
                for j=1:size(ubs_i,2)
                    ub = ubs_i(:,j);
                    overlap = HW2_Utils.rectOverlap(rects_i, ub);
                    isTruePos_i(overlap >= 0.5) = 1;
                end;
                isTruePos{i} = isTruePos_i;
            end
            detScores = cat(2, detScores{:});
            isTruePos = cat(2, isTruePos{:});
            [ap, prec, rec] = ml_ap(detScores, isTruePos, 1);
        end
        
        % Helper function to get training data for training upper body classifier
        % Positive data is feature vectors for annotated upper bodies
        % Negative data is feature vectors for random image patches
        % Inputs:
        %   dataset: either 'train' or 'val'
        % Outputs:
        %   D: d*n data matrix, each column is a HOG feature vector
        %   lb: n*1 label vector, entries are 1 or -1        
        %   imRegs: 64*64*3*n array for n images
        function [D, lb, imRegs] = getPosAndRandomNegHelper(dataset)
            rng(1234); % reset random generator. Keep same seed for repeatability
            load(sprintf('%s/%sAnno.mat', HW2_Utils.dataDir, dataset), 'ubAnno');
            [posD, negD, posRegs, negRegs] = deal(cell(1, length(ubAnno)));            
            
            for i=1:length(ubAnno)
                ml_progressBar(i, length(ubAnno), 'Processing image');
                im = imread(sprintf('%s/%sIms/%04d.jpg', HW2_Utils.dataDir, dataset, i));
                %im = rgb2gray(im);
                ubs = ubAnno{i}; % annotated upper body
                if ~isempty(ubs)
                    [D_i, R_i] = deal(cell(1, size(ubs,2)));
                    for j=1:length(D_i)
                        ub = ubs(:,j);
                        imReg = im(ub(2):ub(4), ub(1):ub(3),:);
                        imReg = imresize(imReg, HW2_Utils.normImSz);
                        D_i{j} = HW2_Utils.cmpFeat(rgb2gray(imReg));
                        R_i{j} = imReg;
                    end 
                    posD{i}    = cat(2, D_i{:});                    
                    posRegs{i} = cat(4, R_i{:});
                end
                
                % sample k random patches; some will be used as negative exampels
                % Choose k sufficiently large to ensure success
                k = 1000;
                [imH, imW,~] = size(im);
                randLeft = randi(imW, [1, k]);
                randTop = randi(imH, [1, k]);
                randSz = randi(min(imH, imW), [1, k]);
                randRects = [randLeft; randTop; randLeft + randSz - 1; randTop + randSz - 1];
                
                % remove random rects that do not lie within image boundaries
                badIdxs = or(randRects(3,:) > imW, randRects(4,:) > imH);
                randRects = randRects(:,~badIdxs);
                
                % Remove random rects that overlap more than 30% with an annotated upper body
                for j=1:size(ubs,2)
                    overlap = HW2_Utils.rectOverlap(randRects, ubs(:,j));                    
                    randRects = randRects(:, overlap < 0.3);
                    if isempty(randRects)
                        break;
                    end;
                end;
                
                % Now extract features for some few random patches
                nNeg2SamplePerIm = 2;
                [D_i, R_i] = deal(cell(1, nNeg2SamplePerIm));
                for j=1:nNeg2SamplePerIm
                    imReg = im(randRects(2,j):randRects(4,j), randRects(1,j):randRects(3,j),:);
                    imReg = imresize(imReg, HW2_Utils.normImSz);
                    R_i{j} = imReg;
                    D_i{j} = HW2_Utils.cmpFeat(rgb2gray(imReg));                    
                end
                negD{i} = cat(2, D_i{:});                
                negRegs{i} = cat(4, R_i{:});
            end    
            posD = cat(2, posD{:});
            negD = cat(2, negD{:});   
            D = cat(2, posD, negD);
            lb = [ones(size(posD,2),1); -ones(size(negD,2), 1)];
            imRegs = cat(4, posRegs{:}, negRegs{:});            
        end;
                
        % rects: 4*k or 5*k matrix for k rectangles, 
        %   rects(1:4,i) is left, top, right, bottom
        function drawRects(rects)            
            rects = double(rects);
            for i=1:size(rects,2)
                box = [rects(1:2,i); rects(3:4,i) - rects(1:2,i) + 1];
                rectangle('Position', box, 'edgecolor', 'g', 'LineWidth', 3);
                if size(rects,1) >= 5
                    text(rects(1,i), rects(2,i), sprintf('%d: %.2f', i, rects(5,i)), ...
                        'Color', 'green', 'FontSize', 14, 'HorizontalAlignment', 'left', ...
                        'VerticalAlignment', 'bottom');
                end;
            end;
        end;
        
        % Compute the symmetric intersection over union overlap between rects set of
        % rects and a single rect
        % rects a 4*k matrix where each column specifies a rectangle
        % a_rect a 4*1 single rectangle for left, top, right, bottom
        function o = rectOverlap(rects, a_rect)
            rects = rects';
            a_rect = a_rect';
            
            x1 = max(rects(:,1), a_rect(1));
            y1 = max(rects(:,2), a_rect(2));
            x2 = min(rects(:,3), a_rect(3));
            y2 = min(rects(:,4), a_rect(4));
            
            w = x2-x1 + 1;
            h = y2-y1 + 1;
            inter = w.*h;
            aarea = (rects(:,3)-rects(:,1) + 1) .* (rects(:,4)-rects(:,2) + 1);
            barea = (a_rect(3)-a_rect(1) + 1) * (a_rect(4)-a_rect(2) + 1);
            % intersection over union overlap
            o = inter ./ (aarea+barea-inter);
            % set invalid entries to 0 overlap
            o(w <= 0) = 0;
            o(h <= 0) = 0;
        end
        
        % Compute feature vector for an image patch
        % Here we use HOG feature, using vl_hog
        function featVec = cmpFeat(imReg)
            featVec = vl_hog(im2single(imReg), HW2_Utils.hogCellSz);            
            featVec = featVec(:);
        end;
        
        % L2 normalization
        % D: d*n matrix for n data points
        function D = l2Norm(D)
            % Add epsilon to avoid division by 0
            D = D./repmat(sqrt(sum(D.^2,1)) + eps, size(D,1), 1);
        end;
        
        % Non-maximum suppression.
        % Greedily select high-scoring detections and skip detections
        % that are significantly covered by a previously selected detection.
        % rects: 5*m rectangles, rects(:,i) is [x1, y1, x2, y2, score]
        % overlap: retained rects must not have intersection/union more than overlap. 
        % top: 5*k retained rects (k <= m).
        function [top, pick] = nms(rects, overlap)
            if isempty(rects)
                pick = [];
                top = [];
            else
                x1 = rects(1,:);
                y1 = rects(2,:);
                x2 = rects(3,:);
                y2 = rects(4,:);
                s  = rects(5,:);
                area = (x2-x1+1) .* (y2-y1+1);
                
                [~, I] = sort(s);
                pick = zeros(1, size(rects,2));
                cnt = 0;                
                while ~isempty(I)
                    last = length(I);
                    i = I(last);
                    cnt = cnt + 1;
                    pick(cnt) = i;
                    
                    %suppress = [last];
                    suppress = false(1, last);
                    suppress(last) = true;
                    for pos = 1:last-1
                        j = I(pos);
                        xx1 = max(x1(i), x1(j));
                        yy1 = max(y1(i), y1(j));
                        xx2 = min(x2(i), x2(j));
                        yy2 = min(y2(i), y2(j));
                        w = xx2-xx1+1;
                        h = yy2-yy1+1;
                        if w > 0 && h > 0
                            % compute overlap
                            o = w * h / area(j);
                            if o > overlap
                                suppress(pos) = true; 
                            end
                        end
                    end
                    I(suppress) = [];
                end
                top = rects(:, pick(1:cnt));
            end            
        end
        
        % Obsolete code
        % Perform sliding window detection and return a list of rectangular regions with scores
        % w, b: weight and bias term of SVM
        % rects: 5*k matrix for k detections, sorted by detection scores
        %   rects(:,i) is left, top, right, bottom, detection score
        % same as detect, but more general and slower. It compute feature vector for each region
        % separately; it does not assume the characteristics of HOG feature image.
        function rects = detect_slow(im, w, b) 
            winSz = HW2_Utils.normImSz;
            stepSz = ceil(HW2_Utils.normImSz/8);
            
            smallestUbSize = 45; % desired smallest upper body size we can detect
            biggestUbSize = 310; % desired biggest  upper body size we can detect

            % To detect upper bodies of multiple sizes, we need to run the detector on the resized
            % image at multiple scales. Here we determine the scales to run detection. 
            % We use log-scale, ie, we increase the size of the image by a constant ratio from one
            % scale to the next.             
            smallestScale = winSz(1)/biggestUbSize;
            biggestScale = winSz(1)/smallestUbSize;            
            nScale = 10;
            scaleStep = (log(biggestScale)-log(smallestScale))/(nScale-1);
            scales = exp(log(smallestScale) + (0:(nScale-1))*scaleStep);
            
            grayIm = rgb2gray(im); % convert to rgb
            
            rects = cell(1, length(scales)); % store the 
            for s = 1:length(scales) % consider each scale in turn
                % Resize the image to a particular scale
                scale = scales(s);
                scaleIm = imresize(grayIm, scale);
                
                % Consider mulple subwindows of size winSz with stepSz (i.e, sliding window)
                % ML_SlideWin is an efficient way to do sliding window. It considers multiple 
                % windows at the same time, avoiding for loop. 
                obj = ML_SlideWin(scaleIm, winSz, stepSz);
                nBatch = obj.getNBatch;
                [topLefts_s, scores_s] = deal(cell(1, nBatch));
                for i=1:nBatch
                    [imRegs, topLefts_s{i}] = obj.getBatch(i);
                    imRegs = reshape(imRegs, winSz(1), winSz(2), size(imRegs,2));
                    D = cell(1, size(imRegs,3));
                    for j=1:size(imRegs,3)
                        D{j} = HW2_Utils.cmpFeat(imRegs(:,:,j));                         
                    end;
                    D = cat(2, D{:});
                    D = HW2_Utils.l2Norm(D);
                    scores_s{i} = D'*w + b;
                end;
                scores_s   = cat(1, scores_s{:});
                topLefts_s = cat(2, topLefts_s{:});
                rects_s = [topLefts_s; topLefts_s + repmat(winSz', 1, size(topLefts_s,2))];
                rects_s = rects_s/scale;
                rects{s} = [rects_s; scores_s'];
            end
            rects = cat(2, rects{:});    
            rects = HW2_Utils.nms(rects, 0.5); 
            
        end;
    end    
end

