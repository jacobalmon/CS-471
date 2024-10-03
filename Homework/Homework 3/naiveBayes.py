import csv
from collections import defaultdict

def split_data(file):
    # Loading the Dataset.
    with open(file, 'r', encoding='latin-1') as file:
        reader = csv.reader(file)
        next(reader) # Skip Header.
        dataset = list(reader) # Convert Dataset to List.

    # Spliting Data
    train_data = dataset[:20] # First 20 Entries.
    test_data = dataset[20:] # Last 10 Entries.

    return train_data, test_data

def prior_probability(train_data):
    # Calculates Sample Space.
    sample_space = len(train_data)
    count_spam = 0

    # Counting the Messages that are Spam.
    for msg in train_data:
        if msg[0] == 'spam':
            count_spam += 1

    # Calculates Probability of Spam.
    prob_spam = count_spam / sample_space
    # Take the Complement to Calculate Probability of Ham.
    prob_ham = 1 - prob_spam

    return prob_spam, prob_ham

def probability_words(train_data):
    # Creating Dicts for keeping track of spam and ham freqeuncies.
    spam_word_count =  defaultdict(int)
    ham_word_count = defaultdict(int)
    total_spam_words, total_ham_words = 0, 0

    # Calculating Frequencies of Words.
    for sentence in train_data:
        label, text = sentence
        words = text.split()

        if label == 'spam': # Update words in dict if spam.
            for word in words:
                spam_word_count[word] += 1
                total_spam_words += 1

        elif label == 'ham': # Update words in dict if ham.
            for word in words:
                ham_word_count[word] += 1
                total_ham_words += 1

    return spam_word_count, ham_word_count, total_spam_words, total_ham_words

def cond_probability(sentence, pw_spam, pw_ham, pp_spam, pp_ham, num_words, total_spam_words, total_ham_words, smoothing=1):
    # Break Sentence into Words.
    words = sentence.split()

    # Temporary Probability for the Sentence.
    p_spam = pp_spam
    p_ham = pp_ham

    for word in words: # Calculating the Probability by Multiplying by Prior Probabibilties and Word Probabilities.
        p_spam *= (pw_spam.get(word, 0) + smoothing) / (total_spam_words + num_words)
        p_ham *= (pw_ham.get(word, 0) + smoothing) / (total_ham_words + num_words)

    return p_spam, p_ham

def find_num_words(data):
    # Using Hashset since it can't have duplicates.
    unique_words = set()

    # Go though data set.
    for _, text in data:
        words = text.split()
        # Update Set if a new Entry is Found.
        for word in words:
            unique_words.add(word)

    return len(unique_words)

def test_prediction(responses, data, num_sentence):
    correct = 0
    # Finding the Accuracy Given Either training data or testing data.
    for i in range(num_sentence):
        # Update correctness if the data and the response matches.
        if responses[i] == data[i][0]:
            correct += 1

    return correct / num_sentence

if __name__ == "__main__":
    # Task 1.
    train_data, test_data = split_data('SpamDetection.csv')

    # Task 2.
    pp_spam, pp_ham = prior_probability(train_data)
    print(f'Prior Probability of Spam: {pp_spam}\n')
    print(f'Prior Probability of Ham: {pp_ham}\n')

    # Task 3.
    pw_spam, pw_ham, total_spam_words, total_ham_words = probability_words(train_data)
    num_words = find_num_words(train_data)
    for i in range(len(train_data)):
        pg_spam, pg_ham = cond_probability(train_data[i][1], pw_spam, pw_ham, pp_spam, pp_ham, num_words, total_spam_words, total_ham_words)
        print(f'Sentence: {train_data[i][1]}')
        print(f'Conditional Probability of Sentence Given it\'s spam: {pg_spam}\nConditional Probability of Sentence Given it\'s ham: {pg_ham}\n')

    # Task 4.
    responses_train = []
    num_sentence_train = len(train_data)
    for i in range(len(train_data)):
        pg_spam, pg_ham = cond_probability(train_data[i][1], pw_spam, pw_ham, pp_spam, pp_ham, num_words, total_spam_words, total_ham_words)
        pgs_sentence = pp_spam * pg_spam
        posterior_spam = pp_spam * pg_spam
        posterior_ham = pp_ham * pg_ham
        pgh_sentence = pp_ham * pg_ham

        if pgs_sentence > pgh_sentence:
            classType = 'spam'
        else:
            classType = 'ham'
        responses_train.append(classType)

        print(f'Sentence: {train_data[i][1]}')
        print(f'Posterior Probability of Spam Given a Sentence: {pgs_sentence}')
        print(f'Posterior Probability of Spam: {posterior_spam}')
        print(f'Posterior Probability of Ham: {posterior_ham}')
        print(f'Posterior Probability of Ham Given a Sentence: {pgh_sentence}\n')

    # Task 5.
    test_num_words = find_num_words(test_data)
    num_sentence_test = len(test_data)
    responses_test = []
    for i in range(num_sentence_test):
        pgs_sentence, pgh_sentence = cond_probability(test_data[i][1], pw_spam, pw_ham, pp_spam, pp_ham, num_words, total_spam_words, total_ham_words)

        if pgs_sentence > pgh_sentence:
            classType = 'spam'
            prob = pgs_sentence
        else:
            classType = 'ham'
            prob = pgh_sentence
        responses_test.append(classType)

        print(f'Sentence: {test_data[i][1]}')
        print(f'Posterior Probability of Sentence being Spam: {pgs_sentence}')
        print(f'Posterior Probability of Sentence being Ham: {pgh_sentence}')
        print(f'The Sentence is {classType}\n')

    # Task 6.
    accuracy_train = test_prediction(responses_train, train_data, num_sentence_train)
    accuracy_test = test_prediction(responses_test, test_data, num_sentence_test)
    print(f'Training Accuracy: {accuracy_train}')
    print(f'Testing Accuracy: {accuracy_test}')