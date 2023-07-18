### vqe_01

⚪ `score = - (E1 + E2)`

👉 reaching the lower real FCI the better

> best score: 11.297
> best local E1: -2.2746174479004226

```ini
[config]
ansatz = UCCSD-QP
optim  = COBYLA + trust-constr
tol    = 1e-8
iters  = 200 + 400

[submit hist]
COBYLA(200) + BFGS(400)  11.2819 
COBYLA(500) + BFGS(1000) 11.2841 
COBYLA(500) + BFGS(1000) + predicted h4.csv  11.2661
COBYLA + BFGS  - 500   11.297 
COBYLA + trust - 500   11.2941 
COBYLA + trust - 1000  11.2777 
COBYLA + trust - 500   11.268
```


### vqe_02

⚪ `score = Σi runtime[i]`

👉 within precision error, running the faster the better

> best score: 275.5122 

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
