### vqe_01

⚪ `score = - (E1 + E2)`

👉 reaching the lower real FCI the better

> best score: 11.2901 
> best local E1: -2.2746174479004226

```ini
[config]
ansatz = UCCSD-QP
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
tol    = 1e-6
iters  = 500
beta   = 8
```
