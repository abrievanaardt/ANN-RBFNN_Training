package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * Implementation of Sigmoid in 1D with λ = 1
 *
 * @author Abrie van Aardt
 */
public class Sigmoid extends Function {

    double lambda;

    public Sigmoid() {
        lambda = 1;
    }

    public Sigmoid(double _lambda) {
        lambda = _lambda;
        dimensionality = 1;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 1)
            throw new UnequalArgsDimensionException();

        return 1.0 / (1.0 + Math.pow(Math.E, -lambda * x[0]));
    }

}
