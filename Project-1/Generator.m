close all 
clear
clc

%Generate the next possible words
textData = fileread('Robin_hood.txt');
generatedText = generateText(textData, 25, 'robin');
disp(generatedText);


function generatedText = generateText(textData, nWords, startWord)

% Convert the text to lowercase and split into words
textData = lower(textData);
words = strsplit(textData);

% Get the unique words and their counts
[uniqueWords,~,wordIdx] = unique(words);
wordCounts = accumarray(wordIdx,1);

% Get the bigram matrix
bigramMat = zeros(length(uniqueWords));
for i = 1:length(words)-1
    word1Idx = find(strcmp(words{i}, uniqueWords));
    word2Idx = find(strcmp(words{i+1}, uniqueWords));
    bigramMat(word1Idx, word2Idx) = bigramMat(word1Idx, word2Idx) + 1;
end
bigramMat = bsxfun(@rdivide, bigramMat, sum(bigramMat, 2));

% Find the index of the starting word
startWordIdx = find(strcmp(startWord, uniqueWords));

% Generate text using bigram probabilities
generatedText = startWord;
for i = 1:nWords-1
    nextWordProbs = bigramMat(startWordIdx,:);
    nextWordIdx = find(cumsum(nextWordProbs) >= rand(), 1);
    generatedText = [generatedText ' ' uniqueWords{nextWordIdx}];
    startWordIdx = nextWordIdx;
end

end



