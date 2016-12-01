package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * This serves as the services contract for all benchmark functions, including
 * statically defined functions such as {@link Alpine}, as well as functions
 * with configurable neutrality (e.g. stacking plates) and ultimately
 * {@link NetworkError}, for calculation of NN classification error.
 *
 * @author Abrie van Aardt
 */
public abstract class Function {

    public abstract double evaluate(double... x) throws UnequalArgsDimensionException;

    public String getName() {
        return this.getClass().getSimpleName();
    }

    public void setDimensionality(int dim) {
        dimensionality = dim;
    }

    public int getDimensionality() {
        return dimensionality;
    }

    protected int dimensionality;
}
