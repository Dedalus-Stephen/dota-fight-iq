import pymorphy3 as pymorphy2

# 1. Initialize Russian Morphology
morph = pymorphy2.MorphAnalyzer()

# 2. Custom Dota-aware Sentiment Dictionary
# 1.0 = Positive, -1.0 = Very Toxic, 0 = Neutral
lexicon = {
    "хуесос": -1.0,
    "блядский": -0.8,
    "ткупоговловой": -0.7, # Handled as 'тупоголовый' below
    "тупоголовый": -0.7,
    "чайник": -0.3,       # Gaming slang for noob
    "gg": 0.3,            # Usually positive
    "молодец": 0.5,       # Positive (unless sarcastic)
    "обзываться": -0.2,
    "курьер": 0.0,
    "динозавр": 0.0,
    "орчид": 0.0,
    "скил": 0.1,
}

all_word_counts = {
    "чайник": 1, "самовар": 1, "это": 1, "за": 1, "динозавра": 1, 
    "и": 1, "курьера": 1, "не": 3, "жмит": 1, "на": 2, "меня": 1, 
    "третий": 1, "скил": 1, "свой": 1, "блядский": 1, "а": 2, 
    "орчид": 1, "можно": 1, "тебя": 1, "вешать": 1, "я": 3, 
    "тебе": 2, "виверне": 1, "ткупоговловой": 1, "молжецй": 1, 
    "обзывайся": 1, "хуесос": 1, "gg": 1
}

def get_sentiment(word_counts):
    total_score = 0
    meaningful_words = 0
    
    print(f"{'WORD':<15} | {'LEMMA':<15} | {'SCORE'}")
    print("-" * 40)

    for word, count in word_counts.items():
        # Clean word and get its base form (e.g. 'динозавра' -> 'динозавр')
        lemma = morph.parse(word.lower())[0].normal_form
        
        # Check our dictionary
        score = lexicon.get(lemma, 0)
        
        # Special check for known typo in your list
        if "ткупоговловой" in word: score = -0.8
        if "молжецй" in word: score = 0.5 # assumed 'молодец'
        
        if score != 0:
            total_score += (score * count)
            meaningful_words += count
            print(f"{word:<15} | {lemma:<15} | {score:>5}")

    if meaningful_words == 0:
        return 0
    
    return total_score / meaningful_words

# Run the analysis
final_score = get_sentiment(all_word_counts)

print("-" * 40)
print(f"FINAL SENTIMENT SCORE: {final_score:.2f}")

if final_score < -0.2:
    print("MATCH STATUS: TOXIC 🤬")
elif final_score > 0.2:
    print("MATCH STATUS: POSITIVE 😊")
else:
    print("MATCH STATUS: NEUTRAL 😐")