import numpy as np
import importlib.util
import inspect
import time

spec_o = importlib.util.spec_from_file_location('orig', 'src/exp2.py')
mod_orig = importlib.util.module_from_spec(spec_o)
spec_o.loader.exec_module(mod_orig)

spec_n = importlib.util.spec_from_file_location('opt', 'src/exp2_numba_auto.py')
mod_opt = importlib.util.module_from_spec(spec_n)
spec_n.loader.exec_module(mod_opt)

def generate_inputs_for(fn):
    import numpy as np
    import inspect

    sig = inspect.signature(fn)
    n_args = len(sig.parameters)
    name = fn.__name__.lower()

    if n_args == 2:
        return (np.random.randn(80, 60), np.random.randn(80, 60))

    if n_args == 1:
        if "row" in name:
            return (np.random.randn(120, 40),)
        if "norm" in name or "signal" in name:
            return (np.random.randn(5000),)
        return (np.random.randn(2000),)

    return tuple(np.random.randn(2000) for _ in range(n_args))

print('=== BENCHMARK AUTOMATIQUE ===')

f_py = getattr(mod_orig, 'accumulate_matrix')
f_nb = getattr(mod_opt, 'accumulate_matrix')
args = generate_inputs_for(f_py)
f_nb(*args)
t0 = time.perf_counter(); f_py(*args); t_py = time.perf_counter() - t0
t0 = time.perf_counter(); f_nb(*args); t_nb = time.perf_counter() - t0
print('Speedup =', t_py/t_nb)

f_py = getattr(mod_orig, 'sum_rows')
f_nb = getattr(mod_opt, 'sum_rows')
args = generate_inputs_for(f_py)
f_nb(*args)
t0 = time.perf_counter(); f_py(*args); t_py = time.perf_counter() - t0
t0 = time.perf_counter(); f_nb(*args); t_nb = time.perf_counter() - t0
print('Speedup =', t_py/t_nb)

f_py = getattr(mod_orig, 'normalize_signal')
f_nb = getattr(mod_opt, 'normalize_signal')
args = generate_inputs_for(f_py)
f_nb(*args)
t0 = time.perf_counter(); f_py(*args); t_py = time.perf_counter() - t0
t0 = time.perf_counter(); f_nb(*args); t_nb = time.perf_counter() - t0
print('Speedup =', t_py/t_nb)

