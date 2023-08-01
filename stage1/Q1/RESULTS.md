### vqe_01

⚪ `score = - (E1 + E2)`

👉 reaching the lower real FCI the better

> best score: 11.297 (stage1) / 11.2956 (stage2)
> best local E1: -2.2746177087120563

```ini
[config]
ansatz = UCCSD-QP
optim  = COBYLA + BFGS
tol    = 1e-8
iters  = 1000
```


### vqe_02

⚪ `score = Σi runtime[i]`

👉 within precision error, running the faster the better

> best score: 275.5122 (stage1) / 397.1758 (stage2)

```python
config1 = {
  'ansatz':  'QUCC',
  'trotter': 2,
  'optim':   'BFGS',
  'tol':     1e-8,
  'dump':    False,
  'maxiter': 250,
  'debug':   False,
}
config2 = {
  'ansatz':  'QUCC',
  'trotter': 2,
  'optim':   'BFGS',
  'tol':     1e-8,
  'beta':    1,
  'eps':     2e-6,
  'maxiter': 300,
  'debug':   False,
  'cont_evolve': True,    # NOTE: trick
}
```
