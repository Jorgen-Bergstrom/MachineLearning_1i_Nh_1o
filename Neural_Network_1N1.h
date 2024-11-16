// Feedforward Neural Network: 1 input, N hidden, 1 output

#pragma once

#include <string>
#include <vector>
#include "Neural_Network_Settings.h"

enum class ActivationType {
    ReLU,
    sigmoid,
    linear,
    Leaky_ReLU
};

// this function is used by the non-linear optimization solver (nlopt.hpp)
extern double calc_fitness(const std::vector<double> &xin, std::vector<double> &grad, void *data);


class Neural_Network_1N1 {
public:
    Neural_Network_1N1();

    void init(
        int nrHiden_in,
        int method,
        ActivationType act_hidden,
        ActivationType act_output);

    void fit(
        const std::vector<double> &x,
        const std::vector<double> &target,
        const Neural_Network_Settings & settings);

    std::vector<double> prediction(const std::vector<double> &x);

    void print_params();
    void print_err_to_file(const std::string &fname);
    void save_predictions_to_file(const std::vector<double> &xin, const std::string &fname);
    void print_vec(const std::string &name, const std::vector<double> &vec);

    double activation(double x, ActivationType act);
    double activation_der(double x, ActivationType act);

    std::vector<int> create_random_batchset(int N, int batch_size);

    std::vector<double> params; // vector with all network parameters: w1[], b2[], w2[], b2

    std::vector<int> nrFuncEvals;
    std::vector<double> err;

    ActivationType activation_hidden;
    ActivationType activation_output;

    std::vector<double> x, target; // training data
    int nrHidden;
};
