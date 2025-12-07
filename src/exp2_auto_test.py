import numpy as np
import importlib.util
import inspect

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

print('=== TEST AUTOMATIQUE ===')

f_py = getattr(mod_orig, 'accumulate_matrix')
f_nb = getattr(mod_opt, 'accumulate_matrix')
args = generate_inputs_for(f_py)
out_py = f_py(*args)
out_nb = f_nb(*args)
print('OK parité' if np.allclose(out_py, out_nb, atol=1e-6) else 'PARITE NON OK')

f_py = getattr(mod_orig, 'sum_rows')
f_nb = getattr(mod_opt, 'sum_rows')
args = generate_inputs_for(f_py)
out_py = f_py(*args)
out_nb = f_nb(*args)
print('OK parité' if np.allclose(out_py, out_nb, atol=1e-6) else 'PARITE NON OK')

f_py = getattr(mod_orig, 'normalize_signal')
f_nb = getattr(mod_opt, 'normalize_signal')
args = generate_inputs_for(f_py)
out_py = f_py(*args)
out_nb = f_nb(*args)
print('OK parité' if np.allclose(out_py, out_nb, atol=1e-6) else 'PARITE NON OK')

