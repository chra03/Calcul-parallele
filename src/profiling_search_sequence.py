import cProfile
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_sequence_simple import search_sequence_python

data = np.random.randint(0, 10, size=200_000).astype(np.uint8)
seq  = np.array([3,5], dtype=np.uint8)

profiler = cProfile.Profile()
profiler.runcall(search_sequence_python, data, seq)
profiler.print_stats(sort='cumtime')
