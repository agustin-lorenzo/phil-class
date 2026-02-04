import pandas as pd

def is_valid(str):
    str = str.strip()
    return (str != "\n" and
            str != str.upper() and
            not str.startswith("##") and
            not str.isdigit()
            and len(str) != 1)

def create_entries(file, entry_size=300):
    """
    Cleans original .txt file lines and
    creates entries with a word count of 'entry_size' for the dataset.
    
    :param file: Location of the .txt file.
    :param entry_size: Number of words in each entry (default of 300)
    """
    entries = []

    words = []
    f = open(file)
    for line in f:
        if is_valid(line):
            words.extend(line.split())
    
    entry = ""
    for i in range(len(words) - 1):
        if i != 0 and i % entry_size == 0:
            entries.append(entry.rstrip())
            entry = ""
        entry += words[i] + " "
    
    return len(entries), entries


all_entries = []
all_labels = []

def create_series(file, label, entry_size):
    """
    Adds all entires and their labels to lists used later for saving dataset
    
    :param file: Location of .txt file.
    :param label: Label of philosophy for .txt file.
    :param entry_size: Number of words in each entry (default of 300)
    """
    num_entries, entries = create_entries(file, entry_size)
    all_entries.extend(entries)
    all_labels.extend([label] * num_entries)

phils = ["existentialism", "nihilism", "stoicism", "utilitarianism"]
for p in phils:
    create_series(f"data/{p}.txt", p, 300)

df = pd.DataFrame({
    "text": all_entries,
    "labels": all_labels
})

df.to_csv("data/data.csv")