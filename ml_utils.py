import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import re

class GrammarClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.classes = ['arithmetic', 'programming', 'natural_language']
        self.model_path = 'grammar_classifier.pkl'
        
        # Load pre-trained model if it exists
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self._train_initial_model()
    
    def _train_initial_model(self):
        # Sample training data
        training_data = [
            # Arithmetic examples
            "E -> E + T | E - T | T",
            "T -> T * F | T / F | F",
            "F -> ( E ) | number",
            
            # Programming examples
            "stmt -> if ( expr ) stmt | while ( expr ) stmt | id = expr",
            "expr -> expr + term | expr - term | term",
            "term -> term * factor | term / factor | factor",
            
            # Natural language examples
            "S -> NP VP",
            "NP -> Det N | N",
            "VP -> V NP | V"
        ]
        
        labels = ['arithmetic'] * 3 + ['programming'] * 3 + ['natural_language'] * 3
        
        # Train the model
        X = self.vectorizer.fit_transform(training_data)
        self.classifier.fit(X, labels)
        
        # Save the model
        self.save_model()
    
    def predict_grammar_type(self, grammar_text):
        """Predict the type of grammar from the input text."""
        X = self.vectorizer.transform([grammar_text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        return prediction, confidence
    
    def save_model(self):
        """Save the trained model to disk."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load the trained model from disk."""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']

class GrammarErrorDetector:
    """A class to detect common errors in grammar specifications."""
    
    def __init__(self):
        self.error_patterns = {
            'missing_arrow': r'^[A-Z][a-zA-Z]*\s+[^->]',  # Non-terminal without ->
            'empty_production': r'->\s*$',  # Empty right side
            'missing_pipe': r'[^|]\s+[a-zA-Z]+\s+[^|]',  # Missing | between alternatives
            'terminal_as_nonterminal': r'->\s+[a-z][a-zA-Z]*\s+[A-Z]',  # Terminal used as non-terminal
        }
        
        self.error_messages = {
            'missing_arrow': "Missing '->' after non-terminal",
            'empty_production': "Empty production rule",
            'missing_pipe': "Missing '|' between alternatives",
            'terminal_as_nonterminal': "Terminal used as non-terminal",
        }
        
        # Load common grammar patterns for validation
        self.common_patterns = self._load_common_patterns()
    
    def _load_common_patterns(self):
        """Load common grammar patterns for validation."""
        return {
            'arithmetic': {
                'operators': ['+', '-', '*', '/', '(', ')'],
                'terminals': ['number', 'id'],
                'structure': ['expr', 'term', 'factor']
            },
            'programming': {
                'keywords': ['if', 'while', 'for', 'return'],
                'terminals': ['id', 'number', 'string'],
                'structure': ['stmt', 'expr', 'decl']
            },
            'natural_language': {
                'parts_of_speech': ['NP', 'VP', 'Det', 'N', 'V', 'Adj', 'Adv'],
                'terminals': [],
                'structure': ['S', 'NP', 'VP']
            }
        }
    
    def detect_errors(self, grammar_text):
        """Detect errors in the grammar specification."""
        errors = []
        lines = grammar_text.strip().split('\n')
        
        # Check each line for errors
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for pattern-based errors
            for pattern_name, pattern in self.error_patterns.items():
                if re.search(pattern, line):
                    errors.append({
                        'line': i + 1,
                        'type': pattern_name,
                        'message': self.error_messages[pattern_name],
                        'suggestion': self._get_suggestion(pattern_name, line)
                    })
            
            # Check for structural errors
            structural_errors = self._check_structural_errors(line, i, lines)
            errors.extend(structural_errors)
        
        # Check for grammar-wide errors
        grammar_wide_errors = self._check_grammar_wide_errors(lines)
        errors.extend(grammar_wide_errors)
        
        return errors
    
    def _check_structural_errors(self, line, line_num, all_lines):
        """Check for structural errors in the grammar."""
        errors = []
        
        # Check for direct left recursion
        if '->' in line:
            lhs, rhs = line.split('->', 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # Check for direct left recursion
            if rhs.startswith(lhs):
                # This is a valid pattern in some grammars, not an error
                pass
        
        return errors
    
    def _check_grammar_wide_errors(self, lines):
        """Check for errors that span multiple lines or the entire grammar."""
        errors = []
        
        # Extract all non-terminals and terminals
        non_terminals = set()
        terminals = set()
        
        for line in lines:
            if '->' in line:
                lhs, rhs = line.split('->', 1)
                lhs = lhs.strip()
                non_terminals.add(lhs)
                
                # Extract symbols from right side
                rhs = rhs.strip()
                for symbol in re.findall(r'[A-Za-z][a-zA-Z]*', rhs):
                    if symbol[0].isupper():
                        non_terminals.add(symbol)
                    else:
                        terminals.add(symbol)
        
        # Check for undefined non-terminals
        for line in lines:
            if '->' in line:
                lhs, rhs = line.split('->', 1)
                rhs = rhs.strip()
                for symbol in re.findall(r'[A-Z][a-zA-Z]*', rhs):
                    if symbol not in non_terminals and symbol != lhs.strip():
                        errors.append({
                            'line': 'Grammar-wide',
                            'type': 'undefined_nonterminal',
                            'message': f"Undefined non-terminal: {symbol}",
                            'suggestion': f"Add a production rule for {symbol} or correct the spelling"
                        })
        
        # Check for unreachable non-terminals
        reachable = {'S'}  # Start with the start symbol
        changed = True
        while changed:
            changed = False
            for line in lines:
                if '->' in line:
                    lhs, rhs = line.split('->', 1)
                    lhs = lhs.strip()
                    if lhs in reachable:
                        for symbol in re.findall(r'[A-Z][a-zA-Z]*', rhs):
                            if symbol not in reachable:
                                reachable.add(symbol)
                                changed = True
        
        for nt in non_terminals:
            if nt not in reachable:
                errors.append({
                    'line': 'Grammar-wide',
                    'type': 'unreachable_nonterminal',
                    'message': f"Unreachable non-terminal: {nt}",
                    'suggestion': f"Remove {nt} or add a path from the start symbol to {nt}"
                })
        
        return errors
    
    def _get_suggestion(self, error_type, line):
        """Get a suggestion for fixing an error."""
        if error_type == 'missing_arrow':
            return f"Add '->' after the non-terminal: {line.split()[0]} -> ..."
        elif error_type == 'empty_production':
            return "Add a right-hand side to the production rule"
        elif error_type == 'missing_pipe':
            return "Add '|' between alternatives"
        elif error_type == 'terminal_as_nonterminal':
            return "Terminals should not be used as non-terminals"
        else:
            return "Review the grammar rule for correctness" 