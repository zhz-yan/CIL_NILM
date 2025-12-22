## CIL Methods on HSV VI Trajectories with ResNet20
This repository implements five CIL methods (acil, bic, der, icarl, il2a) using HSV VI trajectory features and ResNet20 backbone, built on the [pyCIL](https://github.com/LAMDA-CL/PyCIL) framework.

### Testing Case 1
To quickly test with a specific method, pass the configuration file path via the command line argument:

```Bash
# Template
python main.py --config ./exps/<method_name>.json
```Examples:

Run with BiC:
```Bash
python main.py --config ./exps/bic.json
```
Run with ACIL:
```Bash
python main.py --config ./exps/acil.json
```
(Note: Ensure the corresponding JSON files exist in the ./exps/ directory.)


