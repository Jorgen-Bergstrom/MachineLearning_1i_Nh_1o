
class Neural_Network_Settings {
public:
    Neural_Network_Settings();

    int method; // 1=vanilla, 2=non-linear opt, 3=adam

    // vanilla
    int batch_size;
    int max_epochs;
    double learning_rate;
    double clipval;

    // adam
    double alpha;
    double beta1;
    double beta2;
    double epsilon;

    // non-linear search
    double ftol_rel;
    double ftol_abs;
    double xtol_rel;
    int maxeval;

    int verb;
};
