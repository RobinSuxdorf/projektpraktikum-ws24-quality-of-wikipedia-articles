def buildingLexiconLetters():
    """creates a lexicon with letter encodings"""
    alphabet = "abcdefghijklmnopqurstuvwxyz"
    letterLexicon = {}
    for pos in range(len(alphabet)):
        letterLexicon[alphabet[pos]] = pos + 1

    return letterLexicon
