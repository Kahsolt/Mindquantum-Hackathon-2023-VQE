### vqe_01

⚪ `score = - (E1 + E2)`

👉 reaching the lower real FCI the better

> best score: 11.2901 

```ini
[config]
ansatz = UCCSD
optim  = COBYLA + trust-constr
tol    = 1e-6
iters  = 500
```


### vqe_02

⚪ `score = Σi runtime[i]`

👉 within precision error, running the faster the better

> best score: 275.5122 

```ini
[config]
ansatz = QUCC
optim  = trust-constr
tol    = 1e-5
iters  = 500
beta   = 4
```
