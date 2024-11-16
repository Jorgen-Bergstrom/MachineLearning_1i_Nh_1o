#include "Neural_Network_Settings.h"

Neural_Network_Settings::Neural_Network_Settings()
{
    method = 3; // adam
    batch_size = 1;
    max_epochs = 100;
    learning_rate = 1.0e-4;
    clipval = 100.0; // clip gradients at this magnitude
    alpha = 0.001;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1.0e-8;
    ftol_rel = 1.0e-6;
    ftol_abs = 1.0e-8;
    xtol_rel = 1.0e-8;
    maxeval = 1000;
    verb = 0;
}
