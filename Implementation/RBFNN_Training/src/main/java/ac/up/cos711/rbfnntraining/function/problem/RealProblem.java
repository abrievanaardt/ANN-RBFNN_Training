package ac.up.cos711.rbfnntraining.function.problem;

import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class represents a function, bounded in all dimensions by a hypercube
 * and with a known optimum value. This is intended to be treated as a landscape
 * that can be searched by an optimisation algorithm. Where problems do not have
 * known optima, this is indicated with Double.NaN and the corresponding vector
 * will be null.
 *
 * @author Abrie van Aardt
 * @author Dr Katherine Malan
 */
public abstract class RealProblem extends Function {
    
    public RealProblem(double xmin, double xmax, int dim) {
        this(xmin, xmax, dim, 0);
    }

    public RealProblem(double xmin, double xmax, int dim, double fmin) {
        lowerBound = xmin;
        upperBound = xmax;
        dimensionality = dim;
        optimumFitness = fmin;        
    }  

    public double getLowerBound() {
        return lowerBound;
    }

    public double getUpperBound() {
        return upperBound;
    }
    
    public void setLowerBound(double _lowerBound){
        lowerBound = _lowerBound;
    }
    
    public void setUpperBound(double _upperBound){
        upperBound = _upperBound;
    }

    public double getOptimumFitness() {
        return optimumFitness;
    }

    public void setOptimumFitness(double d) {
        optimumFitness = d;
    }
    protected double lowerBound = -10; // lowerBound & upperBound refer to the domain of the problem
    protected double upperBound = 10;
    protected double optimumFitness = 0;// optimum fitness value 
}
