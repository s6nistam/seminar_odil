# Burgers equation

## L-BFGS without multigrid

```
./burgers.py --optimizer lbfgsb --multigrid 0 --kimp 1 --every_factor 10
```

## L-BFGS with multigrid

```
./burgers.py --optimizer lbfgsb --multigrid 1 --kimp 100 --every_factor 2
```

Output directory `out_burgers`

## Newton

```
./burgers.py --optimizer newton --multigrid 0 --linsolver direct --every_factor 0.01
```

The multigrid decomposition has to be disabled for Newton, since the extra unknowns would make the problem underdetermined.
Newton's method needs newer iterations (`--every_factor` sets the factor for the number of steps).
