package ac.up.cos711.rbfnntraining.function;

import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 * Measures Euclidean distance between points in 2D. This is provided for the 
 * {@link PlateStacker}. If distance in higher dimensions are required, this
 * class can be modified. Parameters are assumed to be in the order of the 
 * points to measure, that is:
 * <br>
 * x1, y1, x2, y2
 *
 * @author Abrie van Aardt
 */
public class Distance extends Function{

    public Distance(){
        dimensionality = 4;
    }

    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != getDimensionality())
            throw new UnequalArgsDimensionException("Distance measure is 4 dimensional: x1, y1, x2, y2");
        
        return Math.sqrt(Math.pow(x[0] - x[2], 2) + Math.pow(x[1] - x[3], 2));
    }   
}
