#!/usr/bin/bash

##
## Vanilla
##
if false; then
    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_01.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_02.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_03.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_04.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_05.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_06.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_07.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_08.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_09.txt

    ./runit 1
    mv NN_err.txt results/NN_err_vanilla_10.txt
fi

##
## Adam
##
if true; then
    ./runit 3
    mv NN_err.txt results/NN_err_adam_01.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_02.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_03.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_04.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_05.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_06.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_07.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_08.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_09.txt

    ./runit 3
    mv NN_err.txt results/NN_err_adam_10.txt
fi


##
## Non-linear Optimization
##
if false; then
    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_01.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_02.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_03.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_04.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_05.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_06.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_07.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_08.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_09.txt

    ./runit 2
    mv NN_err.txt results/NN_err_nonlin_10.txt
fi


echo done.
