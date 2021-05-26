"""Tokenizers for natural language SQL queries, and lambda calculus."""
import nltk
import sqlparse


def nl_tokenize(string):
    """Tokenizes a natural language string into tokens.

    Inputs:
       string: the string to tokenize.
    Outputs:
        a list of tokens.

    Assumes data is space-separated (this is true of ZC07 data in ATIS2/3).
    """
    return nltk.word_tokenize(string)


def sql_tokenize(string):
    """ Tokenizes a SQL statement into tokens.

    Inputs:
       string: string to tokenize.

    Outputs:
       a list of tokens.
    """
    tokens = []
    statements = sqlparse.parse(string)

    # SQLparse gives you a list of statements.
    for statement in statements:
        # Flatten the tokens in each statement and add to the tokens list.
        flat_tokens = sqlparse.sql.TokenList(statement.tokens).flatten()
        for token in flat_tokens:
            strip_token = str(token).strip()
            if len(strip_token) > 0:
                tokens.append(strip_token)

    newtokens = []
    keep = True
    for i, token in enumerate(tokens):
        if token == ".":
            newtoken = newtokens[-1] + "." + tokens[i + 1]
            newtokens = newtokens[:-1] + [newtoken]
            keep = False
        elif keep:
            newtokens.append(token)
        else:
            keep = True

    return newtokens


def lambda_tokenize(string):
    """ Tokenizes a lambda-calculus statement into tokens.

    Inputs:
       string: a lambda-calculus string

    Outputs:
       a list of tokens.
    """

    space_separated = string.split(" ")

    new_tokens = []

    # Separate the string by spaces, then separate based on existence of ( or
    # ).
    for token in space_separated:
        tokens = []

        current_token = ""
        for char in token:
            if char == ")" or char == "(":
                tokens.append(current_token)
                tokens.append(char)
                current_token = ""
            else:
                current_token += char
        tokens.append(current_token)
        new_tokens.extend([tok for tok in tokens if tok])

    return new_tokens
