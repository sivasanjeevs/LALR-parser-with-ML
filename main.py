import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from design import Ui_MainWindow
from ml_utils import GrammarClassifier, GrammarErrorDetector

from impl import calculate_first, term_and_nonterm, get_augmented, find_states, combine_states, get_parse_table
from state import State, lalrState

class Parser(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(852, 671)
        self.setWindowTitle("LALR Parser")

        # Initialize ML classifiers
        self.grammar_classifier = GrammarClassifier()
        self.error_detector = GrammarErrorDetector()

        self.init()

        self.ui.action_Open.triggered.connect(self.open_file)
        self.ui.action_Exit.triggered.connect(self.exit_app)
        self.ui.display.clicked.connect(self.disp)
        self.ui.first.clicked.connect(self.disp_first)
        self.ui.lr1.clicked.connect(self.disp_lr1_states)
        self.ui.lalr.clicked.connect(self.disp_lalr_states)
        self.ui.parse_table.clicked.connect(self.disp_parse_table)
        self.ui.plainTextEdit.textChanged.connect(self.check_changed)
        self.ui.parse.clicked.connect(self.disp_parsing)
        self.ui.actionAuthor.triggered.connect(self.disp_author)
        # Add new buttons for ML features
        self.ui.classify_grammar = QtWidgets.QPushButton("Classify Grammar", self)
        self.ui.classify_grammar.setGeometry(650, 20, 150, 30)
        self.ui.classify_grammar.clicked.connect(self.classify_grammar)
        
        self.ui.check_errors = QtWidgets.QPushButton("Check for Errors", self)
        self.ui.check_errors.setGeometry(650, 60, 150, 30)
        self.ui.check_errors.clicked.connect(self.check_grammar_errors)

    def init(self):
        self.grammar = []
        self.augment_grammar = []
        self.first = {}
        self.term = []
        self.non_term = []
        self.states = []
        self.lalr_states = []
        self.parse_table = []
        State.state_count = -1
        lalrState.state_count = 0

    def check_changed(self):
        self.changed = True

    def open_file(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Grammar file')
        if file:
            with open(file, 'r') as f:
                self.ui.plainTextEdit.setPlainText(f.read())
            self.ui.lineEdit.clear()
            self.ui.textBrowser.clear()

    def read_input(self):
        self.init()
        lines = self.ui.plainTextEdit.toPlainText()
        lines_list = lines.split('\n')

        try:
            for line in lines_list:
                line = line.replace(' ', '')

                if line != '':
                    line_list = line.split('->')

                    if line_list[0].isupper() and line_list[1] != '':
                        if '|' in line_list[1]:
                            prod_list = line_list[1].split('|')
                            for prod in prod_list:
                                self.grammar.append([line_list[0], prod])
                        else:
                            self.grammar.append(line_list)
                    else:
                        self.ui.textBrowser.clear()
                        self.ui.textBrowser.setText("Invalid grammar")
                        self.grammar = []

            if self.grammar != []:
                term_and_nonterm(self.grammar, self.term, self.non_term)
                calculate_first(self.grammar, self.first, self.term, self.non_term)
                get_augmented(self.grammar, self.augment_grammar)
                find_states(self.states, self.augment_grammar, self.first, self.term, self.non_term)
                combine_states(self.lalr_states, self.states)
                get_parse_table(self.parse_table, self.lalr_states, self.augment_grammar)
                self.changed = False

        except (KeyError, IndexError):
            self.ui.textBrowser.clear()
            self.ui.textBrowser.setText("Invalid grammar")
            self.init()

    def disp(self):
        self.ui.textBrowser.clear()
        if self.grammar == [] or self.changed:
            self.read_input()

        if self.grammar != []:
            for prod in self.grammar:
                s = prod[0] + ' -> ' + prod[1] + '\n'
                self.ui.textBrowser.append(s)
            self.ui.textBrowser.append("\nNon Terminals : " + ' '.join(self.non_term) + "\nTerminals : " + ' '.join(self.term))

    def disp_first(self):
        if self.first == {} or self.changed:
            self.read_input()
        if self.first != {}:
            self.ui.textBrowser.clear()
            for nonterm in self.non_term:
                self.ui.textBrowser.append('First(' + nonterm + ') : ' + ' '.join(self.first[nonterm]) + '\n')

    def disp_lr1_states(self):
        if self.states == [] or self.changed:
            self.read_input()
        if self.states != []:
            self.ui.textBrowser.clear()
            self.ui.textBrowser.append("Number of LR(1) states : " + str(self.states[len(self.states) - 1].state_num + 1))
            for state in self.states:
                self.ui.textBrowser.append('----------------------------------------------------------------')
                self.ui.textBrowser.append("\nI" + str(state.state_num) + ' : ' + ('goto ( I' + str(state.parent[0]) + " -> '" + str(state.parent[1]) + "' )" if state.state_num != 0 else '') + '\n')
                for item in state.state:
                    self.ui.textBrowser.append(item[0] + ' -> ' + item[1] + ' ,  [ ' + ' '.join(item[2]) + ' ]')
                if state.actions:
                    self.ui.textBrowser.append('\nActions : ')
                    for k, v in state.actions.items():
                        self.ui.textBrowser.insertPlainText(str(k) + ' -> ' + str(abs(v)) + '\t')

    def disp_lalr_states(self):
        if self.lalr_states == [] or self.changed:
            self.read_input()
        if self.lalr_states != []:
            self.ui.textBrowser.clear()
            self.ui.textBrowser.append("Number of LALR states : " + str(lalrState.state_count))
            for state in self.lalr_states:
                self.ui.textBrowser.append('----------------------------------------------------------------')
                self.ui.textBrowser.append("\nI" + str(state.state_num) + ' : ' + ('goto ( I' + str(state.parent[0]) + " -> '" + str(state.parent[1]) + "' )" if state.state_num != 0 else '') + '\nGot by -> ' + str(state.parent_list))
                for item in state.state:
                    self.ui.textBrowser.append(item[0] + ' -> ' + item[1] + ' ,   [ ' + ' '.join(item[2]) + ' ]')
                if state.actions:
                    self.ui.textBrowser.append('\nActions : ')
                    for k, v in state.actions.items():
                        self.ui.textBrowser.insertPlainText(str(k) + ' -> ' + str(abs(v)) + '\t')

    def disp_parse_table(self):
        if self.grammar == [] or self.changed:
            self.read_input()

        if self.grammar != []:
            self.ui.textBrowser.clear()
            all_symb = self.term + ['$'] + self.non_term
            all_symb.remove('e') if 'e' in all_symb else None

            head = '{0:12}'.format(' ')
            for X in all_symb:
                head += '{0:12}'.format(X)
            self.ui.textBrowser.setText(head + '\n')
            s = '------------' * len(all_symb)
            self.ui.textBrowser.append(s)

            for index, state in enumerate(self.parse_table):
                line = '{0:<12}'.format(index)
                for X in all_symb:
                    action = ""
                    if X in state.keys():
                        action = ('s' if state[X] > 0 else 'r' if state[X] < 0 else 'accept') + str(abs(state[X])) if X in self.term else state[X]
                    line += '{0:<12}'.format(action)
                self.ui.textBrowser.append(line)
                self.ui.textBrowser.append(s)

    def disp_parsing(self):
        if self.grammar == [] or self.changed:
            self.read_input()
        if self.grammar != []:
            self.ui.textBrowser.clear()
            line_input = self.ui.lineEdit.text()
            self.parse(self.parse_table, self.augment_grammar, line_input)

    def parse(self, parse_table, augment_grammar, inpt):
        inpt = list(inpt + '$')
        stack = [0]
        a = inpt[0]
        try:
            head = '{0:40} {1:40} {2:40}'.format("Stack", "Input", "Actions")
            self.ui.textBrowser.setText(head)
            while True:
                string = '\n{0:<40} {1:<40} '.format(str(stack), ''.join(inpt))  # Convert stack to string
                s = stack[-1]
                action = parse_table[s][a]
                if action > 0:
                    inpt.pop(0)
                    stack.append(action)
                    self.ui.textBrowser.append(string + 'Shift ' + a + '\n')
                    a = inpt[0]
                elif action < 0:
                    prod = augment_grammar[-action]
                    if prod[1] != 'e':
                        for _ in prod[1]:
                            stack.pop()
                    t = stack[-1]
                    stack.append(parse_table[t][prod[0]])
                    self.ui.textBrowser.append(string + 'Reduce ' + prod[0] + ' -> ' + prod[1] + '\n')
                elif action == 0:
                    self.ui.textBrowser.append('ACCEPT\n')
                    break
        except KeyError:
            self.ui.textBrowser.append('\n\nERROR\n')

    def classify_grammar(self):
        """Classify the current grammar using ML."""
        if self.grammar == [] or self.changed:
            self.read_input()
        
        if self.grammar != []:
            # Convert grammar to text format
            grammar_text = "\n".join([f"{prod[0]} -> {prod[1]}" for prod in self.grammar])
            
            # Get prediction
            grammar_type, confidence = self.grammar_classifier.predict_grammar_type(grammar_text)
            
            # Display results
            self.ui.textBrowser.clear()
            self.ui.textBrowser.append(f"Grammar Classification Results:")
            self.ui.textBrowser.append(f"Type: {grammar_type.replace('_', ' ').title()}")
            self.ui.textBrowser.append(f"Confidence: {confidence:.2%}")
            
            # Add helpful suggestions based on grammar type
            self.ui.textBrowser.append("\nSuggestions:")
            if grammar_type == 'arithmetic':
                self.ui.textBrowser.append("- Consider adding operator precedence rules")
                self.ui.textBrowser.append("- Make sure to handle parentheses correctly")
            elif grammar_type == 'programming':
                self.ui.textBrowser.append("- Consider adding statement blocks with curly braces")
                self.ui.textBrowser.append("- Add support for variable declarations")
            elif grammar_type == 'natural_language':
                self.ui.textBrowser.append("- Consider adding more complex noun phrases")
                self.ui.textBrowser.append("- Add support for adjectives and adverbs")

    def disp_author(self):
        QtWidgets.QMessageBox.information(self, 'Author', 'Author: Your Name')

    def exit_app(self):
        sys.exit()

    def check_grammar_errors(self):
        """Check the grammar for common errors using ML-based error detection."""
        self.ui.textBrowser.clear()
        
        # Get the current grammar text
        grammar_text = self.ui.plainTextEdit.toPlainText()
        
        if not grammar_text.strip():
            self.ui.textBrowser.append("No grammar entered. Please enter a grammar first.")
            return
        
        # Detect errors
        errors = self.error_detector.detect_errors(grammar_text)
        
        if not errors:
            self.ui.textBrowser.append("No errors detected in the grammar.")
            self.ui.textBrowser.append("The grammar appears to be syntactically correct.")
            return
        
        # Display errors
        self.ui.textBrowser.append("Grammar Error Analysis:")
        self.ui.textBrowser.append("------------------------")
        
        # Group errors by line
        line_errors = {}
        grammar_wide_errors = []
        
        for error in errors:
            if error['line'] == 'Grammar-wide':
                grammar_wide_errors.append(error)
            else:
                if error['line'] not in line_errors:
                    line_errors[error['line']] = []
                line_errors[error['line']].append(error)
        
        # Display line-specific errors
        if line_errors:
            self.ui.textBrowser.append("\nLine-specific errors:")
            for line_num in sorted(line_errors.keys()):
                self.ui.textBrowser.append(f"\nLine {line_num}:")
                for error in line_errors[line_num]:
                    self.ui.textBrowser.append(f"  • {error['message']}")
                    self.ui.textBrowser.append(f"    Suggestion: {error['suggestion']}")
        
        # Display grammar-wide errors
        if grammar_wide_errors:
            self.ui.textBrowser.append("\nGrammar-wide errors:")
            for error in grammar_wide_errors:
                self.ui.textBrowser.append(f"  • {error['message']}")
                self.ui.textBrowser.append(f"    Suggestion: {error['suggestion']}")
        
        # Add a summary
        self.ui.textBrowser.append(f"\nTotal errors found: {len(errors)}")
        self.ui.textBrowser.append("\nPlease fix these errors before parsing.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Parser()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
