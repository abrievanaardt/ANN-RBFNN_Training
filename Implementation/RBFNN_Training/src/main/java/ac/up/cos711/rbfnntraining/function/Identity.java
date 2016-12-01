package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * Implementation of the identity function y = x
 *
 * @author Abrie van Aardt
 */
public class Identity extends Function {

    public Identity(){
        dimensionality = 1;
    }
    
    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != 1)
            throw new UnequalArgsDimensionException();
        return x[0];
    }
}
