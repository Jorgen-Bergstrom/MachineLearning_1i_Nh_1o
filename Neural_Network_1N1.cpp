#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <nlopt.hpp>

#include "Neural_Network_1N1.h"



Neural_Network_1N1::Neural_Network_1N1()
{
    init(5, 1, ActivationType::ReLU, ActivationType::linear);
}


void
Neural_Network_1N1::init(
    int nrHidden_in,
    int method,
    ActivationType act_hidden,
    ActivationType act_output)
{
    nrHidden = nrHidden_in;
    activation_hidden = act_hidden;
    activation_output = act_output;
    int N = 3*nrHidden + 1;

    if (method == 1) {
        // custom w and b init
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> random_w(0, 0.1);
        std::uniform_real_distribution<double> random_b(-0.01, 0.01);
        params.resize(N);
        for (int h=0; h < nrHidden; h++) {
            params[h] = random_w(rng);
            params[h+nrHidden] = random_b(rng);
            params[h+2*nrHidden] = random_w(rng);
        }
        params.back() = random_b(rng);
    } else {
        // This is similar to the keras initializer: HeNormal
        double limit = sqrt(6.0 / N);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> random(-limit, limit);
        params.resize(N);
        for (int h=0; h < nrHidden; h++) {
            params[h] = random(rng);
            params[h+nrHidden] = random(rng);
            params[h+2*nrHidden] = random(rng);
        }
        params.back() = random(rng);
    }
}


void
Neural_Network_1N1::fit(
    const std::vector<double> &x_in, // features
    const std::vector<double> &y_in, // targets
    const Neural_Network_Settings &settings)
{
    x = x_in;
    target = y_in;
    nrFuncEvals.clear();
    err.clear();

    // Back propagation -------------------------------
    if (settings.method == 1 || settings.method == 3) {

        // Note: should also keep track of the best prediction
        if (settings.verb > 0) std::cout << "Fit parameters using back propagation" << std::endl;

        const int Nparam = params.size();
        std::vector<double> gradient(Nparam);

        if (settings.verb > 1) print_vec("  initial param", params);

        const int Nfeat = x.size(); // number of features

        // momentum terms (only used by Adam)
        std::vector<double> momentum(Nparam, 0.0);
        std::vector<double> velocity(Nparam, 0.0);

        for (int epoch=0; epoch < settings.max_epochs; epoch++) {
            if (settings.verb > 0 && epoch % (settings.max_epochs/100) == 1) std::cout << "  epoch=" << epoch << std::endl;

            double err2tot = 0;

            // --- CALC: gradient[] ---
            {
                if (settings.verb > 1) std::cout << "  calc gradient" << std::endl;
                std::vector<double> gradientSum(Nparam, 0.0); // increments of params
                std::vector<double> x1(nrHidden); // output of hidden layer

                // pick a random mini-batch in each epoch
                std::vector<int> batch_set;
                batch_set = create_random_batchset(Nfeat, settings.batch_size);

                // loop through all features in the batch
                for (int f : batch_set) {

                    // calculate prediction-----------------------
                    // sets: x1[], arg, y, prefac
                    for (int h=0; h < nrHidden; h++) {
                        double w1 = params[h];
                        double b1 = params[h + nrHidden];
                        x1[h] = activation(w1 * x[f] + b1, activation_hidden);
                    }
                    double arg = params.back(); // b2
                    for (int h=0; h < nrHidden; h++) {
                        double w2 = params[h + 2*nrHidden];
                        arg += w2 * x1[h];
                    }
                    double y = activation(arg, activation_output);
                    double prefac = (target[f] - y);
                    err2tot += pow(target[f] - y, 2.0);
                    // end of prediction-------------------------------------

                    // backpropagation--------------------------------
                    {
                        // layer 2
                        double tmp = prefac * activation_der(arg, activation_output);
                        for (int h=0; h < nrHidden; h++) {
                            gradientSum[h + 2*nrHidden] += tmp * x1[h]; // dw2
                        }
                        gradientSum.back() += tmp;

                        // layer 1
                        for (int h=0; h < nrHidden; h++) {
                            double w1 = params[h];
                            double b1 = params[h + nrHidden];
                            double tmp2 = tmp * activation_der(w1 * x[f] + b1, activation_hidden);
                            gradientSum[h] += tmp2 * x[f]; // dw1
                            gradientSum[h + nrHidden] += tmp2; // db1
                        }
                    }
                }
                if (settings.verb > 1) std::cout << "  err2tot=" << err2tot << std::endl;

                // get the gradient and clip it
                for (int i=0; i < Nparam; i++) {
                    double val = gradientSum[i] / settings.batch_size;
                    val = std::min(val, settings.clipval);
                    val = std::max(val, -settings.clipval);
                    gradient[i] = -val;
                }
                if (settings.verb > 1) print_vec("  gradient", gradient);
            }

            // --- UPDATE: params[]
            if (settings.verb > 1)  std::cout << "\n  update the parameters" << std::endl;

            if (settings.method == 1) { // vanilla
                for (int i=0; i < Nparam; i++) {
                    params[i] -= settings.learning_rate * gradient[i];
                }
            }
            if (settings.method == 3) { // Adam
                for (int i=0; i < Nparam; i++) {
                    momentum[i] = settings.beta1 * momentum[i] + (1.0 - settings.beta1) * gradient[i]; // update biased first momentum estimate
                    velocity[i] = settings.beta2 * velocity[i] + (1.0 - settings.beta2) * pow(gradient[i], 2); // update biased second raw moment estimate
                    double mhat = momentum[i] / (1.0  - pow(settings.beta1, epoch+1)); // bias-corrected first moment estimate
                    double vhat = velocity[i] / (1.0 - pow(settings.beta2, epoch+1)); // bias-corrected second raw moment estimate
                    params[i] = params[i] - settings.alpha * mhat / (sqrt(vhat) + settings.epsilon); // update params

                    if (settings.verb > 1) std::cout << "  i=" << i << ": m=" << momentum[i] << ", v=" << velocity[i] << ", mhat=" << mhat << ", vhat=" << vhat << ", p=" << params[i] << std::endl;
                    if (settings.verb > 1) std::cout << "  param[" << i << "]=" << params[i] << std::endl;
                }
            }

            // stop if error is acceptable
            double tmp = 0.5 * err2tot / settings.batch_size;

            double old_err = std::numeric_limits<double>::max();
            if (err.size() > 0) old_err = err.back();
            err.push_back(std::min(tmp, old_err));

            if (nrFuncEvals.size() == 0) {
                nrFuncEvals.push_back(settings.batch_size);
            } else {
                int prev = nrFuncEvals.back();
                nrFuncEvals.push_back(prev + settings.batch_size);
            }

            if (tmp < settings.ftol_abs) break;

            if (settings.verb > 0 && epoch % (settings.max_epochs/100) == 1) std::cout << "    err2 = " << err[epoch] << std::endl;
        }
    }


    // Non-linear Opt --------------------------------------------
    if (settings.method == 2) {
        if (settings.verb > 0) std::cout << "   initialize nlopt" << std::endl;
        nlopt::opt opt(nlopt::LN_SBPLX, 3*nrHidden+1);
        opt.set_min_objective(calc_fitness, this);
        opt.set_ftol_rel(settings.ftol_rel);
        opt.set_ftol_abs(settings.ftol_abs);
        opt.set_xtol_rel(settings.xtol_rel);
        opt.set_maxeval(settings.maxeval);

        if (settings.verb > 0) std::cout << "   set the initial guess" << std::endl;
        std::vector<double> x0 = params;
        if (settings.verb > 1) print_vec("   x0", x0);

        if (settings.verb > 0) std::cout << "   start non-linear optimization" << std::endl;
        double opt_val;
        nlopt::result result = opt.optimize(x0, opt_val); // ---------------------------------------

        if (settings.verb > 0) {
            std::cout << "\n   Result: " << result << std::endl;
            std::cout << "   Optimal value: " << opt_val << std::endl;
            std::cout << "   Number of evaluations: " << opt.get_numevals() << std::endl;
        }
    }
}


std::vector<double>
Neural_Network_1N1::prediction(const std::vector<double> &xin)
{
    const int N = xin.size();
    std::vector<double> x1(nrHidden);
    std::vector<double> res(N);

    for (int f=0; f < N; f++) {
        for (int h=0; h < nrHidden; h++) {
            double w1 = params[h];
            double b1 = params[h + nrHidden];
            x1[h] = activation(w1 * xin[f] + b1, activation_hidden);
        }
        double arg = params.back();
        for (int h=0; h < nrHidden; h++) {
            double w2 = params[h + 2*nrHidden];
            arg += w2 * x1[h];
        }
        res[f] = activation(arg, activation_output);
    }
    return res;
}


void
Neural_Network_1N1::print_err_to_file(const std::string &fname)
{
    std::ofstream eFile(fname);
    if (!eFile.is_open()) throw std::runtime_error("Could not open file");
    for (int i=0; i < err.size(); i++) {
        eFile << nrFuncEvals[i] << ", " << err[i] << std::endl;
    }
    eFile.close();
}


void
Neural_Network_1N1::save_predictions_to_file(
    const std::vector<double> &xin,
    const std::string &fname)
{
    std::ofstream oFile(fname);
    if (!oFile.is_open()) throw std::runtime_error("Could not open the file.");

    std::vector<double> res;
    res = prediction(xin);

    const int N = xin.size();
    for (int f=0; f < N; f++) {
        oFile << xin[f] << ", " << target[f] << ", " << res[f] << std::endl;
    }
    oFile.close();
}


void
Neural_Network_1N1::print_vec(const std::string &name, const std::vector<double> &vec)
{
    for (int i=0; i < vec.size(); i++) {
        std::cout << name << "[" << i << "]=" << vec[i] << std::endl;
    }
}


double
Neural_Network_1N1::activation(double x, ActivationType type)
{
    switch (type) {
        case ActivationType::ReLU:
            return std::max(0.0, x);
        case ActivationType::sigmoid:
            return 1.0 / (1.0 + exp(-x));
        case ActivationType::linear:
            return x;
        case ActivationType::Leaky_ReLU:
            return std::max(0.1*x, x);
        default:
            throw std::runtime_error("Unsupported activation type");
    }
}


double
Neural_Network_1N1::activation_der(double x, ActivationType type)
{
    switch (type) {
        case ActivationType::ReLU:
            return (x < 0)? 0 : 1;
        case ActivationType::sigmoid:
            return activation(x,type) * (1.0 - activation(x,type));
        case ActivationType::linear:
            return 1;
        case ActivationType::Leaky_ReLU:
            if (x < 0) return 0.1;
            return 1;
        default:
            throw std::runtime_error("Unsupported activation type");
    }
}


std::vector<int>
Neural_Network_1N1::create_random_batchset(int N, int batch_size)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> random_i(0, N-1);

    std::vector<int> res(batch_size);
    for (int i=0; i < batch_size; i++) {
        res[i] = random_i(rng);
    }
    return res;
}


double
calc_fitness(
    const std::vector<double> &params,
    std::vector<double> &grad,
    void *data)
{
    const bool verb {false};
    if (!grad.empty()) throw std::runtime_error("--gradient needed--");

    Neural_Network_1N1 *nPtr;
    nPtr = (Neural_Network_1N1*) data;


    if (nPtr->nrFuncEvals.size() == 0) {
        nPtr->nrFuncEvals.push_back(1);
    } else {
        int prev = nPtr->nrFuncEvals.back();
        nPtr->nrFuncEvals.push_back(prev + 1);
    }

    bool flag = (nPtr->nrFuncEvals.back() % 100 == 0);
    if (verb && flag) {
        std::cout << "calc_fitness() in" << std::endl;
        std::cout << "\tnumber of function calls: " << nPtr->nrFuncEvals.back() << std::endl;
        std::cout << "\tparams.size=" << params.size() << std::endl;
    }

    nPtr->params = params;
    std::vector<double> y = nPtr->prediction(nPtr->x);
    double err {0};
    for (int i=0; i < y.size(); i++) {
        err += 0.5 * pow(y[i] - nPtr->target[i], 2);
    }
    err /= y.size();

    double old_err = std::numeric_limits<double>::max();
    if (nPtr->err.size() > 0) old_err = nPtr->err.back();
    nPtr->err.push_back(std::min(err, old_err));
    if (verb && flag) std::cout << "\terr=" << err << std::endl;
    return err;
}
