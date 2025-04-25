# KylovSubspaceLibrary
**KrylovSubspaceLibrary** is a Python library that provides scalable, matrix-free Krylov subspace optimization methods for training AI models - designed as a faster and more stable alternative to gradient descent.
This library is built with extensibility, performance, and deep learning integration in mind, offering optimizers, and solvers based on Krylov techniques like Conjugate Gradient (CG)

---

## Features

- [x] **Conjugate Gradient (CG)** method for solving Ax = b
- [x] **Matrix-free solvers** using Hessian-vector products
- [x] **PyTorch-compatible optimizer** 'KrylovCG'
- [ ] Preconditioner support *(coming soon)*
- [ ] GMRES, MINRES solvers *(planned)*
- [ ] CLI training interface with config files *(in progress)*
      
---

## Installation
To install locally:
```bash
git clone https://github.com/austinmcb/KrylovSubspaceLibrary.git
cd KrylovSubspaceLibrary
pip install -e .

```
## Example Usage
[coming soon]

## Roadmap

See GitHub Issues for active development, bugs, and future plans

## Contributing

Author: Austin M. McBurney

- Contributions, ideas, and issues are welcome!
  - Fork the repo and create your feature branch (git checkout -b my-feature)
  - Write code + test
  - Open a pull request

## License

This project is licensed under the MIT License -- see the LICENSE file for details
