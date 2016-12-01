package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * Implementation of Tanh in 1D with a coefficient of 1.7159 to allow for the 
 * active domain and range to be +-1.
 *
 * @author Abrie van Aardt
 */
public class Tanh extends Function {

    double coefficient;

    public Tanh() {
        coefficient = 1.7159;
    }

    public Tanh(double _coefficient) {
        coefficient = _coefficient;
        dimensionality = 1;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 1)
            throw new UnequalArgsDimensionException();

        return coefficient * Math.tanh(x[0]);
    }

}
