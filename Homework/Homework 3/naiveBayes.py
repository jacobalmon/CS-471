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

    pw_spam = {} # Dict to Store Probability on Each Word (Spam).
    for word, freq in spam_word_count.items():
        prob = freq / total_spam_words
        pw_spam[word] = prob

    pw_ham = {} # Dict to Store Probability on Each Word (Ham).
    for word, freq in ham_word_count.items():
        prob = freq / total_ham_words
        pw_ham[word] = prob

    return pw_spam, pw_ham, total_spam_words, total_ham_words

def cond_probability(sentence, pw_spam, pw_ham, pp_spam, pp_ham, num_words, smoothing=1):
    words = sentence.split()

    p_spam = pp_spam
    p_ham = pp_ham

    for word in words:
        p_spam *= (pw_spam.get(word, 0) + smoothing) / (sum(pw_spam.values()) + num_words * smoothing)
        p_ham *= (pw_ham.get(word, 0) + smoothing) / (sum(pw_ham.values()) + num_words * smoothing)

    combined_prob = p_spam + p_ham
    pg_spam = p_spam / combined_prob
    pg_ham = 1 - pg_spam

    return pg_spam, pg_ham

# def print_dict(prob_dict, label):
#     print(f'Conditional Probabilities for {label}:')
#     for word, prob in prob_dict.items():
#         print(f'{word}: {prob}')
#     print('')

if __name__ == "__main__":
    # Task 1.
    train_data, test_data = split_data('SpamDetection.csv') 
    # Task 2.
    pp_spam, pp_ham = prior_probability(train_data) 
    # Task 3.
    pw_spam, pw_ham, total_spam_words, total_ham_words = probability_words(train_data) 
    vocab_size = len(set(word for _, text in train_data for word in text.split()))
    pg_spam, pg_ham = cond_probability(train_data[0][1], pw_spam, pw_ham, pp_spam, pp_ham, vocab_size)
    
    # Testing.
    print(f'Prior Probability of Spam: {pp_spam}\n')
    print(f'Prior Probability of Ham: {pp_ham}\n')
    print(f'Conditional Probability of Sentence Given it\'s spam: {pg_spam}\n')
    print(f'Conditional Probability of Sentence Given it\'s ham: {pg_ham}\n')