package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * Implementation of Gaussian Kernel in 1D
 * 
 * @author Abrie van Aardt
 */
public class Gaussian extends Function {

    public Gaussian() {
        dimensionality = 2;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 2)
            throw new UnequalArgsDimensionException();

        return Math.pow(Math.E, -(Math.pow(x[0], 2) / (2 * Math.pow(x[1], 2))));
    }

}
