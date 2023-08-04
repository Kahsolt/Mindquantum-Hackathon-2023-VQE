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

> best score: 275.5122 (stage1) / [mq] 89.0062 | [qp] 71.1257 (stage2)

```python
runner = ocvqe + qupack + hijack

config1 = {
  # circ
  'ansatz':  'UCCSD-QP-hijack',
  'trotter': 2,
  # optim
  'optim':   'BFGS',
  'tol':     1e-4,
  'maxiter': 500,
  'dump':    False,
  # ham
  #'round_one': 6,
  #'round_two': 6,
  #'trunc_one': 0.001,
  #'trunc_two': 0.002,
  #'compress':  1e-5,
}
config2 = {
  'ansatz':  f'UCCSD-QP-hijack',
  'trotter': 4,
  'optim':   'BFGS',
  'tol':     1e-4,
  'beta':    4,
  'eps':     1e-5,
  'maxiter': 1000,
  'cont_evolve': False,
}
```
