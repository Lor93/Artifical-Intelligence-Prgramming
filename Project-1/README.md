Methods of Language Modeling:

We consider two main types of Language Modeling:

Statistical Language Modelings: Statistical Language Modeling, or Language Modeling, is the development of 
probabilistic models that are able to predict the next word in the sequence given the words that precede. 
Examples such as N-gram language modeling.Neural Language Modeling: Neural network methods are achieving better 
results than classical methods both on standalone language models and when models are incorporated into 
larger models on challenging tasks like speech recognition and machine translation.
In this project, you will explore ** Statistical Language Modeling**
Given a sequence of N-1 words, an N-gram model predicts the most probable word that might follow this sequence. 
It's a probabilistic model that's trained on a corpus of text. Such a model is useful in many Natural Language Processing (NLP) applications including speech recognition, machine translation and predictive text input. An N-gram model is built by counting how often word sequences occur in corpus text and then estimating the probabilities. Since a simple N-gram model has limitations, improvements are often made through various techniques.

The goal of this project is to generate text artificially using the probability of appearance of the different words.
Create a function to read the textfile containing the book
Create a function to get all the document in lower case and extract the words
Create a function to count the number of words
Create a function to get the number of unique words
Create a function to get the number of unique words where you specify as an option the minimum number of characters in the word
Create a function to count the occurrence of each word (uni-gram)
Create a function to get a matrix to get the number of words, based on the previous word. (bi-gram)
Create a function that generates text based on the probability of the words, using the previous word as prior information. 
Create additional functions to improve the automatic text generation process, by adding constraint or other rules for the generation.
Create a function to plot the distribution of the unique words
