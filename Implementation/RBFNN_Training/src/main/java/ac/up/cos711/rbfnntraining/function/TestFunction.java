package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * Implements the Sine function for temporarily testing other features.
 *
 * @author Abrie van Aardt
 */
public class TestFunction extends Function {

    public TestFunction(){
        dimensionality = 2;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != getDimensionality())
            throw new UnequalArgsDimensionException();

        return Math.pow(x[0], 2) + Math.pow(x[1], 2);
    }    

}
