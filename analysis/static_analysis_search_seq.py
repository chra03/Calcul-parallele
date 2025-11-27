import ast
import pathlib

files = [
    "src/search_sequence_simple.py",
    "src/search_sequence_numba_parallel.py"
]

for f in files:
    print("\n=== Analyse de :", f, "===\n")
    
    path = pathlib.Path(f).read_text()
    tree = ast.parse(path)

    class Analyzer(ast.NodeVisitor):
        def __init__(self):
            self.loops = 0
            self.length = len(path.splitlines())
            self.ifs = 0

        def visit_For(self, node):
            self.loops += 1
            self.generic_visit(node)

        def visit_If(self, node):
            self.ifs += 1
            self.generic_visit(node)

    analyzer = Analyzer()
    analyzer.visit(tree)

    print("Nombre de lignes :", analyzer.length)
    print("Nombre de boucles for :", analyzer.loops)
    print("Nombre de if :", analyzer.ifs)
