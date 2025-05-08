clc; % 清除命令行窗口内容
clear; % 清除工作区变量
tic; % 开始计时

numImages = 209; % 定义图像的数量为209

% 调用initEdgeData函数，传入图像数量，获取图像列表、二值图像列表、边缘图、边缘距离图、灰度边缘图
[imgList, binImages, edgeMap, edgeDistMap, grayEdgeMap] = initEdgeData(numImages); 

% 用于存储可能是左边的碎片的索引，初始为空数组
leftCandidates = []; 
% 用于存储可能是右边的碎片的索引，初始为空数组
rightCandidates = []; 
leftIdx = 1; % 左边碎片索引列表的索引，初始为1
rightIdx = 1; % 右边碎片索引列表的索引，初始为1

% 遍历所有图像
for i = 1:numImages
    currentEdge = edgeMap{i}; % 获取当前图像的边缘图
    % 如果当前图像左边边缘的和为180且左边边缘距离大于5
    if sum(currentEdge(:,1)) == 180 && edgeDistMap(i,1) > 5
        leftCandidates(leftIdx) = i; % 将当前图像的索引加入左边候选列表
        leftIdx = leftIdx + 1; % 左边索引加1
        continue; % 继续下一次循环
    end
    % 如果当前图像右边边缘的和为180且右边边缘距离大于5
    if sum(currentEdge(:,2)) == 180 && edgeDistMap(i,2) > 5
        rightCandidates(rightIdx) = i; % 将当前图像的索引加入右边候选列表
        rightIdx = rightIdx + 1; % 右边索引加1
        continue; % 继续下一次循环
    end
end

% 用于存储每张图上下边缘属性类型的列表，初始为空
typeList = []; 
% 用于存储每张图上下边缘距离的列表，初始为空
distList = []; 
% 遍历所有图像
for i = 1:numImages
    % 调用getImageEdgeAttributes函数，获取当前二值图像的边缘属性（类型和距离）
    [t, d] = getImageEdgeAttributes(binImages{i}); 
    typeList = [typeList; t]; % 将边缘属性类型加入typeList
    distList = [distList; d]; % 将边缘属性距离加入distList
end

% 初始化图像池，包含从1到图像总数的所有索引
imagePool = 1:numImages; 
% 遍历左边和右边的候选碎片索引列表
for idx = [leftCandidates, rightCandidates]
    % 调用removeFromPool函数，从图像池中移除候选碎片索引
    imagePool = removeFromPool(imagePool, idx); 
end

% 初始化网格图，大小为22行19列，元素初始值为0
gridMap = zeros(22,19); 
% 初始化每个位置的计数器，大小为22行1列，元素初始值为1
slotCounter = ones(22,1); 
% 设置尝试次数，为图像池数量的30倍
attempts = numel(imagePool) * 30; 

% 当尝试次数大于0时执行循环
while attempts > 0
    tic; % 开始计时
    % 生成一个随机索引，范围是1到图像池的长度
    randIdx = ceil(rand() * numel(imagePool)); 
    currentImg = imagePool(randIdx); % 获取当前图像的索引
    matched = false; % 标记是否匹配，初始为false

    % 遍历左边和右边的候选碎片
    for j = 1:(length(leftCandidates) + length(rightCandidates))
        % 如果j小于等于左边候选碎片的数量
        if j <= length(leftCandidates)
            refIdx = leftCandidates(j); % 获取左边候选碎片的索引
        else
            % 计算右边候选碎片的索引
            refIdx = rightCandidates(j - length(leftCandidates)); 
        end

        % 如果当前图像和参考图像的边缘属性类型全部相同
        % 且边缘属性距离的差值的绝对值都小于3
        if all(typeList(refIdx,:) == typeList(currentImg,:)) && ...
           all(abs(distList(refIdx,:) - distList(currentImg,:)) < 3)
            gridMap(j, slotCounter(j)) = currentImg; % 将当前图像放入网格图的相应位置
            slotCounter(j) = slotCounter(j) + 1; % 相应位置的计数器加1
            % 从图像池中移除当前图像的索引
            imagePool = removeFromPool(imagePool, currentImg); 
            matched = true; % 标记为已匹配
            break; % 跳出循环
        elseif all(abs(distList(refIdx,:) - distList(currentImg,:)) < 8)
            % 显示参考图像和当前图像
            subplot(1,2,1); imshow(imgList{refIdx}); 
            subplot(1,2,2); imshow(imgList{currentImg}); 
            % 弹出输入对话框，让用户判断是否为同一行碎片
            answer = inputdlg({'是否为同一行碎片？输入 1 是，0 否'}, '人工判断', 1, {'0'}); 
            % 将用户输入转换为数值
            if str2double(answer{1}) == 1 
                gridMap(j, slotCounter(j)) = currentImg; % 将当前图像放入网格图的相应位置
                slotCounter(j) = slotCounter(j) + 1; % 相应位置的计数器加1
                % 从图像池中移除当前图像的索引
                imagePool = removeFromPool(imagePool, currentImg); 
                matched = true; % 标记为已匹配
                break; % 跳出循环
            end
        end
    end

    toc; % 结束计时并显示时间
    attempts = attempts - 1; % 尝试次数减1
end

% 第一阶段：从左空白碎片出发构造完整行
leftResult = cell(1,length(leftCandidates)); % 初始化左边结果列表，为元胞数组
rightResult = cell(1,length(rightCandidates)); % 初始化右边结果列表，为元胞数组
leftPool = leftCandidates; % 左边碎片池初始为左边候选碎片列表
rightPool = leftCandidates;  % 右边碎片池初始为左边候选碎片列表（仅初始化，后续会调整）

% 遍历左边和右边的候选碎片
for i = 1:(length(leftCandidates) + length(rightCandidates))
    % 如果i小于等于左边候选碎片的数量
    if i <= length(leftCandidates)
        seed = leftCandidates(i); % 获取左边的种子碎片索引
        sequence = gridMap(i,:); % 获取当前行的碎片序列
        % 调用matchLeftToRight函数，进行从左到右的匹配
        [res, leftPool, rightPool, imagePool] = matchLeftToRight(seed, grayEdgeMap, sequence, leftPool, rightPool, imagePool, gridMap, rightCandidates, typeList, distList); 
        leftResult{i} = res; % 将匹配结果存入左边结果列表
    else
        seed = rightCandidates(i - length(leftCandidates)); % 获取右边的种子碎片索引
        sequence = gridMap(i,:); % 获取当前行的碎片序列
        % 调用matchRightToLeft函数，进行从右到左的匹配
        [res, leftPool, rightPool, imagePool] = matchRightToLeft(seed, grayEdgeMap, sequence, leftPool, rightPool, imagePool, gridMap, leftCandidates, typeList, distList); 
        rightResult{i - length(leftCandidates)} = res; % 将匹配结果存入右边结果列表
    end
end

toc; % 结束计时并显示时间

% 初始化边缘数据
function [imgList, binList, edgeList, distList, grayList] = initEdgeData(imgCount)
    distList = []; % 初始化距离列表
    % 遍历所有图像
    for k = 1:imgCount  
        % 生成图像文件名，格式为3位数字补零
        imgName = sprintf('%03d.bmp', k - 1);  
        imgList{k} = imread(imgName); % 读取图像并存入图像列表
        % 提取图像的第1列和第72列，转换为int16类型，存入灰度列表
        grayList{k} = int16(imgList{k}(:,[1 72])); 
        % 将图像转换为二值图像，存入二值图像列表
        binList{k} = im2bw(imgList{k}, graythresh(imgList{k})); 
        % 提取二值图像的第1列和第72列，存入边缘列表
        edgeList{k} = binList{k}(:,[1 72]); 
        % 调用computeEdgeDistance函数，计算边缘距离，存入距离列表
        distList = [distList; computeEdgeDistance(binList{k})]; 
    end
end

% 从图像池中移除指定图像编号
function reducedPool = removeFromPool(pool, idxToRemove)
    % 如果图像池为空
    if isempty(pool)
        reducedPool = []; % 结果为空
        return; % 返回
    end

    % 从图像池中移除指定索引的图像编号
    reducedPool = pool(pool ~= idxToRemove); 
end

% 计算图像上下边缘的黑白边界类型和偏移量
function [edgeType, edgeDist] = getImageEdgeAttributes(binaryImg)
    [rows, ~] = size(binaryImg); % 获取二值图像的行数
    edgeType = zeros(1,2); % 初始化边缘类型，长度为2
    edgeDist = zeros(1,2); % 初始化边缘距离，长度为2

    % 顶部边界分析
    topLine = binaryImg(1,:); % 获取图像第一行
    topIsBlack = any(topLine == 0); % 判断第一行是否有黑色像素

    % 遍历图像的行
    for i = 1:rows
        rowHasBlack = any(binaryImg(i,:) == 0); % 判断当前行是否有黑色像素
        % 如果顶部是黑色且当前行不是黑色
        if topIsBlack && ~rowHasBlack 
            edgeType(1) = 0; % 设置顶部边缘类型为0
            edgeDist(1) = i - 1; % 设置顶部边缘距离
            break; % 跳出循环
        elseif ~topIsBlack && rowHasBlack % 如果顶部不是黑色且当前行是黑色
            edgeType(1) = 1; % 设置顶部边缘类型为1
            edgeDist(1) = i - 1; % 设置顶部边缘距离
            break; % 跳出循环
        end
    end

    % 底部边界分析
    bottomLine = binaryImg(rows,:); % 获取图像最后一行
    bottomIsBlack = any(bottomLine == 0); % 判断最后一行是否有黑色像素

    % 从最后一行开始遍历
    for j = rows:-1:1
        rowHasBlack = any(binaryImg(j,:) == 0); % 判断当前行是否有黑色像素
        % 如果底部是黑色且当前行不是黑色
        if bottomIsBlack && ~rowHasBlack 
            edgeType(2) = 0; % 设置底部边缘类型为0
            edgeDist(2) = rows - j; % 设置底部边缘距离
            break; % 跳出循环
        elseif ~bottomIsBlack && rowHasBlack % 如果底部不是黑色且当前行是黑色
            edgeType(2) = 1; % 设置底部边缘类型为1
            edgeDist(2) = rows - j; % 设置底部边缘距离
            break; % 跳出循环
        end
    end
end
% 计算图像左右边缘距图像边界的距离（用于判断是否为空白）
function dist = computeEdgeDistance(binaryImg)
    [~, cols] = size(binaryImg); % 获取二值图像的列数

    % 查找左边空白宽度
    for i = 1:cols
        if any(binaryImg(:,i) == 0) % 如果当前列有黑色像素
            dist(1) = i - 1; % 设置左边边缘距离
            break; % 跳出循环
        end
    end

    % 查找右边空白宽度
    for j = cols:-1:1
        if any(binaryImg(:,j) == 0) % 如果当前列有黑色像素
            dist(2) = cols - j; % 设置右边边缘距离
            break; % 跳出循环
        end
    end
end

% 根据边缘像素差异计算误差评分，用于碎片配对评估
function [score1, score2, score3] = calculateErrorDegree(edgeA, edgeB, threshold)
    n = size(edgeA, 1); % 获取边缘A的行数
    errorFlags = zeros(1, 180); % 初始化错误标志数组，长度为180

    % 遍历边缘的行（从第3行到倒数第3行）
    for i = 3:(n-2) 
        % 计算边缘A和边缘B的差值，按照一定权重计算
        diff = ...
            0.7 * (edgeA(i,2) - edgeB(i,1)) + ...
            0.1 * (edgeA(i-1,2) - edgeB(i-1,1)) + ...
            0.1 * (edgeA(i+1,2) - edgeB(i+1,1)) + ...
            0.05 * (edgeA(i-2,2) - edgeB(i-2,1)) + ...
            0.05 * (edgeA(i+2,2) - edgeB(i+2,1)); 

        errorFlags(i) = abs(diff) > threshold; % 如果差值绝对值大于阈值，设置错误标志为1
    end

    % 分段累计误差数量，分别计算三个部分的误差数量
    score1 = sum(errorFlags(1:60)); 
    score2 = sum(errorFlags(61:119)); 
    score3 = sum(errorFlags(120:180)); 
end
% 从左边缘碎片出发，逐步向右拼接整行碎片
function [result, leftPool, rightPool, imagePool] = matchLeftToRight(startIdx, edgeMap, sequence, leftPool, rightPool, imagePool, gridMap, rightEnds, typeList, distList)
    threshold = 25; % 设置误差阈值为25
    currentEdge = edgeMap{startIdx}; % 获取起始碎片的边缘图
    result = startIdx; % 结果初始为起始碎片的索引
    anchor = startIdx; % 锚点初始为起始碎片的索引
    pos = 2; % 位置初始为2

    % 遍历当前行的碎片序列
    for step = 1:nnz(sequence) 
        errorList = []; % 初始化误差列表

        % 与当前行未使用的候选碎片尝试匹配
        for j = 1:nnz(sequence)
            candidate = sequence(j); % 获取候选碎片的索引
            edgeB = edgeMap{candidate}; % 获取候选碎片的边缘图
            % 调用calculateErrorDegree函数，计算误差评分
            [s1, s2, s3] = calculateErrorDegree(currentEdge, edgeB, threshold); 
            % 将相关信息存入误差列表
            errorList = [errorList; anchor, candidate, s1, s2, s3]; 
        end

        % 与图像池中的碎片匹配
        for j = 1:length(imagePool)
            candidate = imagePool(j); % 获取图像池中的候选碎片索引
            edgeB = edgeMap{candidate}; % 获取候选碎片的边缘图
            % 调用calculateErrorDegree函数，计算误差评分
            [s1, s2, s3] = calculateErrorDegree(currentEdge, edgeB, threshold); 
            % 将相关信息存入误差列表
            errorList = [errorList; anchor, candidate, s1, s2, s3]; 
        end

        % 标准化误差并排序
        min1 = min(errorList(:,3)); % 获取误差评分1的最小值
        min2 = min(errorList(:,4)); % 获取误差评分2的最小值
        min3 = min(errorList(:,5)); % 获取误差评分3的最小值
        weightedError = []; % 初始化加权误差列表

        % 计算加权误差
        for i = 1:size(errorList,1)
            if errorList(i,3) ~= 0 % 如果误差评分1不为0
                norm1 = (errorList(i,3) - min1) / errorList(i,3); % 标准化误差评分1
            else
                norm1 = 0; % 否则为0
            end
        
            if errorList(i,4) ~= 0 % 如果误差评分2不为0
                norm2 = (errorList(i,4) - min2) / errorList(i,4); % 标准化误差评分2
            else
                norm2 = 0; % 否则为0
            end
        
            if errorList(i,5) ~= 0 % 如果误差评分3不为0
                norm3 = (errorList(i,5) - min3) / errorList(i,5); % 标准化误差评分3
            else
                norm3 = 0; % 否则为0
            end
        
            % 将相关信息存入加权误差列表
            weightedError = [weightedError; errorList(i,1:2), norm1 + norm2 + norm3]; 
        end


        [~, idxMin] = min(weightedError(:,3)); % 获取加权误差最小的索引
                nextPiece = weightedError(idxMin,2); % 获取最小加权误差对应的碎片索引
        if idxMin <= nnz(sequence) % 如果索引在当前行的碎片序列范围内
            % 匹配自当前序列
            result(pos) = nextPiece; % 将匹配的碎片索引存入结果中
            anchor = nextPiece; % 更新锚点
            currentEdge = edgeMap{nextPiece}; % 更新当前边缘图
            sequence = removeFromPool(sequence, nextPiece); % 从序列中移除已匹配的碎片索引
            pos = pos + 1; % 位置加1
        else
            % 匹配自 imagePool，但需满足类型与距离要求
            prevPiece = result(pos - 1); % 获取上一个碎片索引
            sameType = ...
                typeList(prevPiece,1) == typeList(nextPiece,1) || ...
                typeList(prevPiece,2) == typeList(nextPiece,2); % 判断类型是否相同
            distClose = ...
                abs(distList(prevPiece,1) - distList(nextPiece,1)) < 5 || ...
                abs(distList(prevPiece,2) - distList(nextPiece,2)) < 5; % 判断距离是否接近

            if sameType && distClose % 如果类型相同且距离接近
                result(pos) = nextPiece; % 将匹配的碎片索引存入结果中
                anchor = nextPiece; % 更新锚点
                currentEdge = edgeMap{nextPiece}; % 更新当前边缘图
                imagePool = removeFromPool(imagePool, nextPiece); % 从图像池中移除已匹配的碎片索引
                pos = pos + 1; % 位置加1
            end
        end
    end

    leftPool = removeFromPool(leftPool, startIdx); % 从左边碎片池中移除起始碎片索引

    % 若长度为18，尝试与右侧空白碎片闭合匹配
    if length(result) == 18 
        lastEdge = edgeMap{result(end)}; % 获取最后一个碎片的边缘图
        errorList = []; % 初始化误差列表
        for i = 1:length(rightEnds)
            if gridMap(12:22,1)==0  % 只考虑空位
                rightEdge = edgeMap{rightEnds(i)}; % 获取右边候选碎片的边缘图
                % 调用calculateErrorDegree函数，计算误差评分
                [s1,s2,s3] = calculateErrorDegree(lastEdge, rightEdge, threshold); 
                % 将相关信息存入误差列表
                errorList = [errorList; result(end), rightEnds(i), s1, s2, s3]; 
            end
        end
        if ~isempty(errorList) % 如果误差列表不为空
            [~, idx] = min(errorList(:,3)); % 获取误差最小的索引
            rightEnd = errorList(idx,2); % 获取误差最小的右边碎片索引
            result(pos) = rightEnd; % 将右边碎片索引存入结果中
            rightPool = removeFromPool(rightPool, rightEnd); % 从右边碎片池中移除该碎片索引
        end
    end
end
% 从右侧空白碎片向左构造完整行，方向与 matchLeftToRight 相反
function [result, leftPool, rightPool, imagePool] = matchRightToLeft(startIdx, edgeMap, sequence, leftPool, rightPool, imagePool, gridMap, leftEnds, typeList, distList)
    threshold = 25; % 设置误差阈值为25
    currentEdge = edgeMap{startIdx}; % 获取起始碎片的边缘图
    result = zeros(1,19); % 初始化结果数组，长度为19
    result(19) = startIdx; % 将起始碎片索引存入结果数组的最后一个位置
    anchor = startIdx; % 锚点初始为起始碎片的索引
    pos = 18; % 位置初始为18

    % 遍历当前行的碎片序列
    for step = 1:nnz(sequence) 
        errorList = []; % 初始化误差列表

        % 与当前行未使用的候选碎片尝试匹配
        for j = 1:nnz(sequence)
            candidate = sequence(j); % 获取候选碎片的索引
            edgeA = edgeMap{candidate}; % 获取候选碎片的边缘图
            % 调用calculateErrorDegree函数，计算误差评分
            [s1, s2, s3] = calculateErrorDegree(edgeA, currentEdge, threshold); 
            % 将相关信息存入误差列表
            errorList = [errorList; candidate, anchor, s1, s2, s3]; 
        end

        % 与图像池中的碎片匹配
        for j = 1:length(imagePool)
            candidate = imagePool(j); % 获取图像池中的候选碎片索引
            edgeA = edgeMap{candidate}; % 获取候选碎片的边缘图
            % 调用calculateErrorDegree函数，计算误差评分
            [s1, s2, s3] = calculateErrorDegree(edgeA, currentEdge, threshold); 
            % 将相关信息存入误差列表
            errorList = [errorList; candidate, anchor, s1, s2, s3]; 
        end

        % 归一化误差值
        min1 = min(errorList(:,3)); % 获取误差评分1的最小值
        min2 = min(errorList(:,4)); % 获取误差评分2的最小值
        min3 = min(errorList(:,5)); % 获取误差评分3的最小值
        weightedError = []; % 初始化加权误差列表

        % 计算加权误差
        for i = 1:size(errorList,1)
            if errorList(i,3) ~= 0 % 如果误差评分1不为0
                norm1 = (errorList(i,3) - min1) / errorList(i,3); % 标准化误差评分1
            else
                norm1 = 0; % 否则为0
            end
        
            if errorList(i,4) ~= 0 % 如果误差评分2不为0
                norm2 = (errorList(i,4) - min2) / errorList(i,4); % 标准化误差评分2
            else
                norm2 = 0; % 否则为0
            end
        
            if errorList(i,5) ~= 0 % 如果误差评分3不为0
                norm3 = (errorList(i,5) - min3) / errorList(i,5); % 标准化误差评分3
            else
                norm3 = 0; % 否则为0
            end
        
            % 将相关信息存入加权误差列表
            weightedError = [weightedError; errorList(i,1:2), norm1 + norm2 + norm3]; 
        end


        [~, idxMin] = min(weightedError(:,3)); % 获取加权误差最小的索引
        nextPiece = weightedError(idxMin,1); % 获取最小加权误差对应的碎片索引
        if idxMin <= nnz(sequence) % 如果索引在当前行的碎片序列范围内
            % 从当前序列中选取匹配片段
            result(pos) = nextPiece; % 将匹配的碎片索引存入结果中
            anchor = nextPiece; % 更新锚点
            currentEdge = edgeMap{nextPiece}; % 更新当前边缘图
            sequence = removeFromPool(sequence, nextPiece); % 从序列中移除已匹配的碎片索引
            pos = pos - 1; % 位置减1
        else
            % 尝试从 imagePool 中匹配
            nextTypeOK = ...
                typeList(result(pos+1),1) == typeList(nextPiece,1) || ...
                typeList(result(pos+1),2) == typeList(nextPiece,2); % 判断类型是否相同
            distClose = ...
                abs(distList(result(pos+1),1) - distList(nextPiece,1)) < 5 || ...
                abs(distList(result(pos+1),2) - distList(nextPiece,2)) < 5; % 判断距离是否接近

            if nextTypeOK && distClose % 如果类型相同且距离接近
                result(pos) = nextPiece; % 将匹配的碎片索引存入结果中
                anchor = nextPiece; % 更新锚点
                currentEdge = edgeMap{nextPiece}; % 更新当前边缘图
                imagePool = removeFromPool(imagePool, nextPiece); % 从图像池中移除已匹配的碎片索引
                pos = pos - 1; % 位置减1
            end
        end
    end

    rightPool = removeFromPool(rightPool, startIdx); % 从右边碎片池中移除起始碎片索引

    % 若长度为18，尝试与左侧空白碎片闭合匹配
    if nnz(result) == 18 
        errorList = []; % 初始化误差列表
        for i = 1:length(leftEnds)
            if gridMap(1:11,1) == 0 % 只考虑空位
                edgeA = edgeMap{leftEnds(i)}; % 获取左边候选碎片的边缘图
                % 调用calculateErrorDegree函数，计算误差评分
                [s1,s2,s3] = calculateErrorDegree(edgeA, currentEdge, threshold); 
                % 将相关信息存入误差列表
                errorList = [errorList; leftEnds(i), result(20-nnz(result)), s1, s2, s3]; 
            end
        end
        if ~isempty(errorList) % 如果误差列表不为空
            [~, idx] = min(errorList(:,3)); % 获取误差最小的索引
            result(pos) = errorList(idx,1); % 将左边碎片索引存入结果中
            leftPool = removeFromPool(leftPool, result(pos)); % 从左边碎片池中移除该碎片索引
        end
    end
end
% 根据图像行上下边缘特征对多个图像序列（行）进行排序
function sortedSeq = sortRowsByEdge(seqMatrix, imgData)
    rowCount = size(seqMatrix, 1); % 获取序列矩阵的行数
    sortedSeq = zeros(rowCount, size(seqMatrix,2)); % 初始化排序后的序列矩阵
    edgeRow = cell(1, rowCount); % 初始化边缘行元胞数组，长度为行数

    % 拼接整行图像，用于计算上下边缘
    for i = 1:rowCount
        tempRow = []; % 初始化临时行
        for j = 1:size(seqMatrix,2)
            tempRow = [tempRow, imgData{seqMatrix(i,j)}]; % 拼接每行的图像
        end
        edgeRow{i} = tempRow; % 将拼接后的行存入边缘行数组
    end

    % 获取每行上下边缘特征
    typeList = []; % 初始化类型列表
    distList = []; % 初始化距离列表
    for i = 1:rowCount
        % 调用getImageEdgeAttributes函数，获取边缘属性（类型和距离）
        [t, d] = getImageEdgeAttributes(logical(edgeRow{i})); 
        typeList = [typeList; t]; % 将边缘属性类型加入类型列表
        distList = [distList; d]; % 将边缘属性距离加入距离列表
    end

    % 找出最上方的白边行作为起始
    topWhite = max(distList(:,1)); % 获取顶部边缘距离的最大值
    for i = 1:rowCount
        if distList(i,1) == topWhite % 如果当前行顶部边缘距离等于最大值
            anchorIdx = i; % 设置锚点索引为当前行索引
            break; % 跳出循环
        end
    end

    % 提取上下边缘图像片段用于匹配
    topBottomEdges = cell(1,rowCount); % 初始化上下边缘元胞数组，长度为行数
    for i = 1:rowCount
        topEdge = []; % 初始化顶部边缘
        bottomEdge = []; % 初始化底部边缘
        for j = 1:size(seqMatrix,2)
            topEdge = [topEdge, imgData{seqMatrix(i,j)}(1,:)]; % 提取每行图像的第一行
            bottomEdge = [bottomEdge, imgData{seqMatrix(i,j)}(end,:)]; % 提取每行图像的最后一行
        end
        topBottomEdges{i} = [topEdge; bottomEdge]; % 将顶部和底部边缘存入上下边缘数组
    end

    % 匹配排序
    threshold = 25; % 设置误差阈值为25
    remaining = 1:rowCount; % 初始化剩余行索引数组
    remaining = removeFromPool(remaining, anchorIdx); % 从剩余行索引数组中移除锚点行索引
    sortedSeq(1,:) = seqMatrix(anchorIdx,:); % 将锚点行存入排序后的序列矩阵的第一行
    refEdge = topBottomEdges{anchorIdx}; % 设置参考边缘为锚点行的上下边缘

    currentIdx = anchorIdx; % 当前索引初始为锚点索引
    for i = 2:rowCount
        errorList = []; % 初始化误差列表
        for j = 1:length(remaining)
            candidateIdx = remaining(j); % 获取候选行索引
            % 调用calculateErrorDegree函数，计算误差评分
            errScore = calculateErrorDegree(refEdge, topBottomEdges{candidateIdx}, threshold); 
            % 将相关信息存入误差列表
            errorList = [errorList; currentIdx, candidateIdx, errScore]; 
        end
        [~, idx] = min(errorList(:,3)); % 获取误差最小的索引
        bestMatch = errorList(idx,2); % 获取误差最小的候选行索引
        sortedSeq(i,:) = seqMatrix(bestMatch,:); % 将误差最小的行存入排序后的序列矩阵
        currentIdx = bestMatch; % 更新当前索引
        refEdge = topBottomEdges{bestMatch}; % 更新参考边缘为误差最小的行的上下边缘
        remaining = removeFromPool(remaining, bestMatch); % 从剩余行索引数组中移除已匹配的行索引
    end
end
% 将编号矩阵对应的图像依序拼接并显示出来（行列拼接）
function drawSequence(idMatrix)
    [rows, cols] = size(idMatrix); % 获取编号矩阵的行数和列数
    fullImage = []; % 初始化完整图像

    for i = 1:rows
        rowImg = []; % 初始化当前行图像
        for j = 1:cols
            id = idMatrix(i,j); % 获取当前图像编号
            fileName = sprintf('%03d.bmp', id - 1);  % 生成文件名，自动补零到3位数
            rowImg = [rowImg, imread(fileName)]; % 读取图像并拼接当前行图像
        end
        fullImage = [fullImage; rowImg]; % 拼接当前行图像到完整图像
    end

    imshow(fullImage); % 显示完整图像
end
% 将一维碎片编号序列拼接显示，用于预览拼图效果
function shredPreview(seq)
    seq = seq(seq ~= 0);  % 去除零元素
    fullImg = []; % 初始化完整图像

    for i = 1:length(seq)
        id = seq(i); % 获取当前碎片编号
        if id <= 10 % 如果编号小于等于10
            fileName = ['00', num2str(id - 1), '.bmp']; % 生成文件名
        elseif id <= 100 % 如果编号小于等于100
            fileName = ['0', num2str(id - 1), '.bmp']; % 生成文件名
        else % 其他情况
            fileName = [num2str(id - 1), '.bmp']; % 生成文件名
        end
        fullImg = [fullImg, imread(fileName)]; % 读取图像并拼接完整图像
    end

    imshow(fullImg); % 显示完整图像
end