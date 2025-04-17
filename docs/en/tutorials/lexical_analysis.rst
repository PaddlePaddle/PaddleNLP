========================
Lexical Analysis
========================

Lexical Analysis (词法分析)
--------------------------

Lexical analysis is the process of converting a character sequence into a token sequence. It is the first phase of compilation and also the foundation of natural language processing.

Main Tasks:
1. Tokenization (分词)
2. Removing whitespace/comments
3. Tracking line numbers
4. Generating symbol tables

Example code:
```python
# This is a comment
x = 123
if (x > 0) {
    print("Positive")
}
```

Mathematical Representation:
$$ T = Lex(S) $$
Where $S$ is the input string and $T$ is the output token sequence.

See details in [link](#lexer-implementation). For HTML tags: <div class="important"></div>