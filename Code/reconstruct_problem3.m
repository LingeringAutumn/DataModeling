% 加载图片 -> 特征提取 -> 初步分类 -> 行拼接 -> 可视化
clc; clear;
tic;

totalCount = 209;  % 图像总数
[imgSet, binSet, edgeSet, edgeDistSet, graySet] = preprocessImages(totalCount);  % 图像边缘预处理

% 初步识别左右边缘为空的碎片
leftEnds = [];
rightEnds = [];
leftPtr = 1; rightPtr = 1;

for idx = 1:totalCount
    edge = edgeSet{idx};
    if sum(edge(:,1)) == 180 && edgeDistSet(idx,1) > 5
        leftEnds(leftPtr) = idx;
        leftPtr = leftPtr + 1;
        continue;
    end
    if sum(edge(:,2)) == 180 && edgeDistSet(idx,2) > 5
        rightEnds(rightPtr) = idx;
        rightPtr = rightPtr + 1;
        continue;
    end
end

% 提取上下边缘黑白类型 + 特征距离
typeMap = [];
distMap = [];
for idx = 1:totalCount
    [type, dist] = extractEdgeInfo(binSet{idx});
    typeMap = [typeMap; type];
    distMap = [distMap; dist];
end

% 去掉左右边碎片，构造初始图像池
shredPool = 1:totalCount;
for outId = [leftEnds, rightEnds]
    shredPool = removeShred(shredPool, outId);
end

% 初步按上下边缘属性分类碎片到候选行
rowGrid = zeros(22,19);
rowFillPtr = ones(22,1);
maxTry = numel(shredPool) * 30;

while maxTry > 0
    tic;
    randIdx = ceil(rand() * numel(shredPool));
    currentId = shredPool(randIdx);
    matched = false;

    for r = 1:(length(leftEnds) + length(rightEnds))
        if r <= length(leftEnds)
            refId = leftEnds(r);
        else
            refId = rightEnds(r - length(leftEnds));
        end
    
        % 自动匹配：上下边缘类型相同，距离足够接近
        if all(typeMap(refId,:) == typeMap(currentId,:)) && ...
           all(abs(distMap(refId,:) - distMap(currentId,:)) < 3)
    
            rowGrid(r, rowFillPtr(r)) = currentId;
            rowFillPtr(r) = rowFillPtr(r) + 1;
            shredPool = removeShred(shredPool, currentId);
            matched = true;
            break;
    
        % 半自动匹配：上下边缘距离差在阈值内，人工确认
        elseif all(abs(distMap(refId,:) - distMap(currentId,:)) < 8)
            subplot(1,2,1); imshow(imgSet{refId});
            subplot(1,2,2); imshow(imgSet{currentId});
            res = inputdlg({'是否为同一行碎片？输入 1 是，0 否'}, '人工确认', 1, {'0'});
            if str2double(res{1}) == 1
                rowGrid(r, rowFillPtr(r)) = currentId;
                rowFillPtr(r) = rowFillPtr(r) + 1;
                shredPool = removeShred(shredPool, currentId);
                matched = true;
                break;
            end
        end
    end

    toc;
    maxTry = maxTry - 1;
end
% 第二阶段：从左右端碎片分别出发拼接整行

leftResults = cell(1, length(leftEnds));
rightResults = cell(1, length(rightEnds));
leftQueue = leftEnds;
rightQueue = leftEnds;  % 起始时两侧都用左端点启动（RightToLeft会换方向）

for r = 1:(length(leftEnds) + length(rightEnds))
    if r <= length(leftEnds)
        seed = leftEnds(r);
        candidateRow = rowGrid(r,:);
        [res, leftQueue, rightQueue, shredPool] = matchLeftToRight(...
            seed, graySet, candidateRow, leftQueue, rightQueue, shredPool, ...
            rowGrid, rightEnds, typeMap, distMap);
        leftResults{r} = res;
    else
        seed = rightEnds(r - length(leftEnds));
        candidateRow = rowGrid(r,:);
        [res, leftQueue, rightQueue, shredPool] = matchRightToLeft(...
            seed, graySet, candidateRow, leftQueue, rightQueue, shredPool, ...
            rowGrid, leftEnds, typeMap, distMap);
        rightResults{r - length(leftEnds)} = res;
    end
end

toc;
function [imgList, binList, edgeList, distList, grayList] = preprocessImages(count)
% 图像预处理：读取 000a.bmp ~ 208b.bmp 共两类图像
% 输出多个列表：原图、二值图、灰度边缘、边缘提取、空白边宽度

    distList = [];
    imgIdx = 1;

    for id = 0:(count - 1)
        for suffix = ['a', 'b']
            fileName = sprintf('%03d%c.bmp', id, suffix);

            if ~isfile(fileName)
                error('图像文件不存在：%s', fileName);
            end

            img = imread(fileName);
            imgList{imgIdx} = img;

            % 提取灰度边缘
            grayList{imgIdx} = int16(img(:, [1 72]));

            % 二值化图像
            bin = im2bw(img, graythresh(img));
            binList{imgIdx} = bin;

            % 左右边缘（用于误差匹配）
            edgeList{imgIdx} = bin(:, [1 72]);

            % 左右空白宽度
            distList = [distList; measureEdgeWhiteSpace(bin)];

            imgIdx = imgIdx + 1;
        end
    end
end

function distance = measureEdgeWhiteSpace(binaryImg)
% 计算图像左右边缘的空白像素宽度

    [~, cols] = size(binaryImg);

    % 左边缘
    for left = 1:cols
        if any(binaryImg(:, left) == 0)
            distance(1) = left - 1;
            break;
        end
    end

    % 右边缘
    for right = cols:-1:1
        if any(binaryImg(:, right) == 0)
            distance(2) = cols - right;
            break;
        end
    end
end
function updatedPool = removeShred(pool, toRemove)
% 从碎片池中移除指定编号

    if isempty(pool)
        updatedPool = [];
        return;
    end

    % 如果碎片池中只有一个元素，直接判断是否要移除
    if length(pool) == 1
        updatedPool = [];
        return;
    end

    % 遍历保留非目标元素
    temp = [];
    for k = 1:length(pool)
        if pool(k) == toRemove
            continue;
        end
        temp = [temp, pool(k)];
    end
    updatedPool = temp;
end
function [edgeType, edgeOffset] = extractEdgeInfo(binaryImg)
% 提取图像上下边缘的黑白类型与边界偏移量
% edgeType: 0 表示黑到白，1 表示白到黑
% edgeOffset: 从上下边缘向中心的边界偏移量（单位：像素行数）

    [rows, ~] = size(binaryImg);
    edgeType = zeros(1, 2);
    edgeOffset = zeros(1, 2);

    % 顶部边缘判断
    topRow = binaryImg(1, :);
    topIsBlack = any(topRow == 0);

    for i = 1:rows
        rowBlack = any(binaryImg(i, :) == 0);
        if topIsBlack && ~rowBlack
            edgeType(1) = 0;
            edgeOffset(1) = i - 1;
            break;
        elseif ~topIsBlack && rowBlack
            edgeType(1) = 1;
            edgeOffset(1) = i - 1;
            break;
        end
    end

    % 底部边缘判断
    bottomRow = binaryImg(rows, :);
    bottomIsBlack = any(bottomRow == 0);

    for i = rows:-1:1
        rowBlack = any(binaryImg(i, :) == 0);
        if bottomIsBlack && ~rowBlack
            edgeType(2) = 0;
            edgeOffset(2) = rows - i;
            break;
        elseif ~bottomIsBlack && rowBlack
            edgeType(2) = 1;
            edgeOffset(2) = rows - i;
            break;
        end
    end
end
function [score1, score2, score3] = computeErrorScore(edgeA, edgeB, threshold)
% 计算两个边缘之间的误差分段得分
% 返回：3段误差评分，用于评估左右边缘是否可拼接

    n = size(edgeA, 1);
    diffFlags = zeros(1, 180);

    for i = 3:(n - 2)
        diff = ...
            0.7 * (edgeA(i,2) - edgeB(i,1)) + ...
            0.1 * (edgeA(i-1,2) - edgeB(i-1,1)) + ...
            0.1 * (edgeA(i+1,2) - edgeB(i+1,1)) + ...
            0.05 * (edgeA(i-2,2) - edgeB(i-2,1)) + ...
            0.05 * (edgeA(i+2,2) - edgeB(i+2,1));

        if abs(diff) > threshold
            diffFlags(i) = 1;
        end
    end

    % 分段累加误差点数量
    score1 = sum(diffFlags(1:60));
    score2 = sum(diffFlags(61:119));
    score3 = sum(diffFlags(120:180));
end
function [resultRow, leftPool, rightPool, remainingPool] = matchLeftToRight(...
    seedId, edgeMap, rowTemplate, leftPool, rightPool, remainingPool, ...
    gridRef, rightEnds, typeMap, distMap)
% 从左边界碎片出发，依次向右匹配补全一行碎片
% 输入为一行候选编号，输出为拼接完成的一行碎片编号

    threshold = 25;
    currentEdge = edgeMap{seedId};
    resultRow = seedId;
    anchorId = seedId;
    col = 2;

    for step = 1:nnz(rowTemplate)
        errList = [];

        % 先尝试与当前 rowTemplate 中未使用的碎片比较
        for j = 1:nnz(rowTemplate)
            candidate = rowTemplate(j);
            edgeB = edgeMap{candidate};
            [s1, s2, s3] = computeErrorScore(currentEdge, edgeB, threshold);
            errList = [errList; anchorId, candidate, s1, s2, s3];
        end

        % 再尝试与池中剩余碎片匹配
        for j = 1:length(remainingPool)
            candidate = remainingPool(j);
            edgeB = edgeMap{candidate};
            [s1, s2, s3] = computeErrorScore(currentEdge, edgeB, threshold);
            errList = [errList; anchorId, candidate, s1, s2, s3];
        end

        % 归一化误差并排序
        min1 = min(errList(:,3));
        min2 = min(errList(:,4));
        min3 = min(errList(:,5));
        weightedErrors = [];

        for i = 1:size(errList,1)
            if errList(i,3) ~= 0
                norm1 = (errList(i,3) - min1) / errList(i,3);
            else
                norm1 = 0;
            end
            if errList(i,4) ~= 0
                norm2 = (errList(i,4) - min2) / errList(i,4);
            else
                norm2 = 0;
            end
            if errList(i,5) ~= 0
                norm3 = (errList(i,5) - min3) / errList(i,5);
            else
                norm3 = 0;
            end
            weightedErrors = [weightedErrors; errList(i,1:2), norm1 + norm2 + norm3];
        end

        [~, minIdx] = min(weightedErrors(:,3));
        nextId = weightedErrors(minIdx, 2);
                if minIdx <= nnz(rowTemplate)
            % 如果选中的是原始行模板中的碎片
            resultRow(col) = nextId;
            anchorId = nextId;
            currentEdge = edgeMap{nextId};
            rowTemplate = removeShred(rowTemplate, nextId);
            col = col + 1;
        else
            % 如果选中的是剩余池中的碎片，需进一步检查匹配条件
            prevId = resultRow(col - 1);
            isTypeMatch = ...
                typeMap(prevId,1) == typeMap(nextId,1) || ...
                typeMap(prevId,2) == typeMap(nextId,2);
            isDistMatch = ...
                abs(distMap(prevId,1) - distMap(nextId,1)) < 5 || ...
                abs(distMap(prevId,2) - distMap(nextId,2)) < 5;

            if isTypeMatch && isDistMatch
                resultRow(col) = nextId;
                anchorId = nextId;
                currentEdge = edgeMap{nextId};
                remainingPool = removeShred(remainingPool, nextId);
                col = col + 1;
            end
        end
    end

    % 从左池中移除当前起点碎片
    leftPool = removeShred(leftPool, seedId);

    % 如果拼到了第18个位置，尝试封闭右端
    if length(resultRow) == 18
        rightEdge = edgeMap{resultRow(end)};
        finalMatches = [];

        for i = 1:length(rightEnds)
            if gridRef(12:22,1) == 0  % 判断是否空位（此条件可能需细化）
                edgeB = edgeMap{rightEnds(i)};
                [s1,s2,s3] = computeErrorScore(rightEdge, edgeB, threshold);
                finalMatches = [finalMatches; resultRow(end), rightEnds(i), s1, s2, s3];
            end
        end

        if ~isempty(finalMatches)
            [~, matchIdx] = min(finalMatches(:,3));
            matchedRight = finalMatches(matchIdx,2);
            resultRow(col) = matchedRight;
            rightPool = removeShred(rightPool, matchedRight);
        end
    end
end
function [resultRow, leftPool, rightPool, remainingPool] = matchRightToLeft(...
    seedId, edgeMap, rowTemplate, leftPool, rightPool, remainingPool, ...
    gridRef, leftEnds, typeMap, distMap)
% 从右侧边缘碎片出发，向左拼接一行碎片

    threshold = 25;
    currentEdge = edgeMap{seedId};
    resultRow = zeros(1, 19);
    resultRow(19) = seedId;
    anchorId = seedId;
    col = 18;

    for step = 1:nnz(rowTemplate)
        errList = [];

        % 遍历当前行模板中未使用的碎片
        for j = 1:nnz(rowTemplate)
            candidate = rowTemplate(j);
            edgeA = edgeMap{candidate};
            [s1, s2, s3] = computeErrorScore(edgeA, currentEdge, threshold);
            errList = [errList; candidate, anchorId, s1, s2, s3];
        end

        % 遍历剩余池中的碎片
        for j = 1:length(remainingPool)
            candidate = remainingPool(j);
            edgeA = edgeMap{candidate};
            [s1, s2, s3] = computeErrorScore(edgeA, currentEdge, threshold);
            errList = [errList; candidate, anchorId, s1, s2, s3];
        end

        % 归一化误差值
        min1 = min(errList(:,3));
        min2 = min(errList(:,4));
        min3 = min(errList(:,5));
        weightedErrors = [];

        for i = 1:size(errList,1)
            if errList(i,3) ~= 0
                norm1 = (errList(i,3) - min1) / errList(i,3);
            else
                norm1 = 0;
            end
            if errList(i,4) ~= 0
                norm2 = (errList(i,4) - min2) / errList(i,4);
            else
                norm2 = 0;
            end
            if errList(i,5) ~= 0
                norm3 = (errList(i,5) - min3) / errList(i,5);
            else
                norm3 = 0;
            end
            weightedErrors = [weightedErrors; errList(i,1:2), norm1 + norm2 + norm3];
        end

        [~, minIdx] = min(weightedErrors(:,3));
        nextId = weightedErrors(minIdx, 1);
                if minIdx <= nnz(rowTemplate)
            % 如果来自模板行
            resultRow(col) = nextId;
            anchorId = nextId;
            currentEdge = edgeMap{nextId};
            rowTemplate = removeShred(rowTemplate, nextId);
            col = col - 1;
        else
            % 如果来自剩余碎片池，验证类型与距离
            nextTo = resultRow(col + 1);  % 当前右侧的碎片编号
            isTypeMatch = ...
                typeMap(nextTo,1) == typeMap(nextId,1) || ...
                typeMap(nextTo,2) == typeMap(nextId,2);
            isDistMatch = ...
                abs(distMap(nextTo,1) - distMap(nextId,1)) < 5 || ...
                abs(distMap(nextTo,2) - distMap(nextId,2)) < 5;

            if isTypeMatch && isDistMatch
                resultRow(col) = nextId;
                anchorId = nextId;
                currentEdge = edgeMap{nextId};
                remainingPool = removeShred(remainingPool, nextId);
                col = col - 1;
            end
        end
    end

    rightPool = removeShred(rightPool, seedId);

    % 如果当前拼了18个碎片，尝试拼接左端边界碎片完成闭合
    if nnz(resultRow) == 18
        finalMatches = [];

        for i = 1:length(leftEnds)
            if gridRef(1:11,1) == 0  % 检查空位（此判断需按实际位置优化）
                edgeA = edgeMap{leftEnds(i)};
                [s1,s2,s3] = computeErrorScore(edgeA, currentEdge, threshold);
                finalMatches = [finalMatches; leftEnds(i), resultRow(col + 1), s1, s2, s3];
            end
        end

        if ~isempty(finalMatches)
            [~, bestIdx] = min(finalMatches(:,3));
            matchedLeft = finalMatches(bestIdx,1);
            resultRow(col) = matchedLeft;
            leftPool = removeShred(leftPool, matchedLeft);
        end
    end
end
function sortedRows = sortRowsByEdge(rowSeq, imgData)
% 对多行碎片序列进行排序，使上下边缘更连续
% 输入：rowSeq 每行是若干拼接好的碎片编号
% 输出：sortedRows 排序后的行顺序

    rowCount = size(rowSeq, 1);
    sortedRows = zeros(rowCount, size(rowSeq,2));
    rowImages = cell(1, rowCount);

    % 拼接每一行的图像（用于后续边缘分析）
    for r = 1:rowCount
        rowImg = [];
        for c = 1:size(rowSeq,2)
            rowImg = [rowImg, imgData{rowSeq(r,c)}];
        end
        rowImages{r} = rowImg;
    end

    % 提取上下边缘黑白类型与偏移
    typeMap = [];
    distMap = [];
    for r = 1:rowCount
        [t, d] = extractEdgeInfo(logical(rowImages{r}));
        typeMap = [typeMap; t];
        distMap = [distMap; d];
    end

    % 找出顶部白边最多的行，作为初始基准行
    topWhiteMax = max(distMap(:,1));
    for r = 1:rowCount
        if distMap(r,1) == topWhiteMax
            anchorRow = r;
            break;
        end
    end

    % 提取上下边缘用于误差匹配
    edgeBlocks = cell(1,rowCount);
    for r = 1:rowCount
        topEdge = [];
        bottomEdge = [];
        for c = 1:size(rowSeq,2)
            block = imgData{rowSeq(r,c)};
            topEdge = [topEdge, block(1,:)];
            bottomEdge = [bottomEdge, block(end,:)];
        end
        edgeBlocks{r} = [topEdge; bottomEdge];
    end

    % 按边缘误差迭代排序
    threshold = 25;
    candidates = 1:rowCount;
    candidates = removeShred(candidates, anchorRow);
    sortedRows(1,:) = rowSeq(anchorRow,:);
    current = anchorRow;
    currentEdge = edgeBlocks{current};

    for i = 2:rowCount
        errList = [];
        for j = 1:length(candidates)
            target = candidates(j);
            score = computeErrorScore(currentEdge, edgeBlocks{target}, threshold);
            errList = [errList; current, target, score];
        end
        [~, bestIdx] = min(errList(:,3));
        nextRow = errList(bestIdx, 2);
        sortedRows(i,:) = rowSeq(nextRow,:);
        currentEdge = edgeBlocks{nextRow};
        candidates = removeShred(candidates, nextRow);
    end
end
function drawGridSequence(gridMatrix)
% 将二维碎片编号矩阵拼接为整图并显示
% gridMatrix 是 m×n 的碎片编号，编号从 1 开始

    [rows, cols] = size(gridMatrix);
    combinedImage = [];

    for r = 1:rows
        rowImage = [];
        for c = 1:cols
            id = gridMatrix(r,c);
            if id == 0
                continue;  % 跳过空位
            end
            fileName = sprintf('%03d.bmp', id - 1);
            block = imread(fileName);
            rowImage = [rowImage, block];
        end
        if ~isempty(rowImage)
            combinedImage = [combinedImage; rowImage];
        end
    end

    imshow(combinedImage);
end
function previewShredRow(seq)
% 展示一行碎片的拼接图像（用于预览单行拼图效果）

    seq = seq(seq ~= 0);  % 移除空位
    combinedRow = [];

    for i = 1:length(seq)
        id = seq(i);
        fileName = sprintf('%03d.bmp', id - 1);
        block = imread(fileName);
        combinedRow = [combinedRow, block];
    end

    imshow(combinedRow);
end
function updatedPool = addToPool(pool, value)
% 向碎片池中添加编号，避免重复

    if isempty(pool)
        updatedPool = value;
        return;
    end

    if ~ismember(value, pool)
        updatedPool = [pool, value];
    else
        updatedPool = pool;
    end
end
function normalized = normalizeImage(imageBlock)
% 对图像块按灰度范围进行归一化处理（线性拉伸）

    imageBlock = double(imageBlock);
    minVal = min(imageBlock(:));
    maxVal = max(imageBlock(:));

    if maxVal == minVal
        normalized = zeros(size(imageBlock));  % 避免除以零
    else
        normalized = (imageBlock - minVal) / (maxVal - minVal);
    end
end



