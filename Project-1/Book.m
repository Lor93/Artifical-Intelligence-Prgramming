close all 
clear
clc

%Read the book
book = fileread('Robin_hood.txt');
book = splitlines(book);
book(1,:);
book(3:27) = [];  %Remove extra header
book(11160:end) = [];  %Remove extra footer
disp(book(1:11159));

%Build the bi-gram model 
processed = book;
delimiters = {' ', '!', '''', ',', '-', '.',... 
    ':', ';', '?', '\r', '\n', '--', '&'};
biMdl = bigramClass(delimiters);                
biMdl.build(processed);   
%This get the probablity of any two word that you chose  
row = strcmp(biMdl.unigrams, 'Little');          
col= strcmp(biMdl.unigrams, 'John');             
biMdl.mdl(row,col)   
rng(1)

%Clean the String array
TF = (book == " ");
book(TF) = [];
book(1,:)

%Replacing or Remove Punctuation
Punc = [",", ".", "?", "!", ";", " ", ":", " "" " ];
book = replace(book, Punc, " ");
book(1,:)

%Stripping leading and tail space character
book= strip(book);
book(1,:)
bookWords = string(0);
for i=1:length(book)
    bookWords = [bookWords ; split(book(i))];
end
bookWords(1,:)

%Finding unique word
bookWords = lower(bookWords);
[book,~,idx] = unique(bookWords);
numOccurrences = histcounts(idx, numel(book));
[rankOfOccurrences, rankIndex] = sort(numOccurrences, 'descend');
wordsByFrequency = book(rankIndex);

%Collect Statistics and Plot the word
numOccurrences = numOccurrences(rankIndex);
numOccurrences = numOccurrences';
numWords = length(bookWords);
T = table;
T.Words = wordsByFrequency;
T.NumOccurrences = numOccurrences;
T.PercentOfText = numOccurrences / numWords * 100.0;
T.CumulativePercentOfText = cumsum(numOccurrences) / numWords * 100.0;

loglog(rankOfOccurrences);
xlabel('Rank of word (most to least common)');
ylabel('Number of Occurrences');






