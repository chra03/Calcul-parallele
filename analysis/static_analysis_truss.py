import ast
import pathlib

FILE = "src/truss.py"

source = pathlib.Path(FILE).read_text()
tree = ast.parse(source)

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.loops = 0
        self.ifs = 0
        self.calls = 0
        self.length = len(source.splitlines())

    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.ifs += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.calls += 1
        self.generic_visit(node)

an = Analyzer()
an.visit(tree)

print("Analyse du fichier:", FILE)
print("Lignes :", an.length)
print("Boucles for :", an.loops)
print("Conditions if :", an.ifs)
print("Appels de fonction :", an.calls)
