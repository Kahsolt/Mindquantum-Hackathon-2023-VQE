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

> best score: 275.5122 (stage1) / 102.8464 (stage2)

```python
runner = ocvqe

config1 = {
  'ansatz':  'QUCC',
  'trotter': 1,
  'optim':   'BFGS',
  'tol':     1e-3,
  'maxiter': 100,
  'dump':    False,
}
config2 = {
  'ansatz':  'QUCC',
  'trotter': 2,
  'optim':   'BFGS',
  'tol':     1e-3,
  'beta':    2,
  'eps':     2e-6,
  'maxiter': 300,
  'cont_evolve': False,
}
```
