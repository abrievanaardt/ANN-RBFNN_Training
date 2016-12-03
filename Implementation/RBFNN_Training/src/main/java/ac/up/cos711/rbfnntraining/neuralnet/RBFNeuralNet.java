package ac.up.cos711.rbfnntraining.neuralnet;

import ac.up.cos711.rbfnntraining.function.Gaussian;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import java.util.Arrays;

/**
 * Default implementation of an {@link IRBFNeuralNet}.
 *
 * @author Abrie van Aardt
 */
public class RBFNeuralNet implements IRBFNeuralNet {

    /**
     * Adds 1 additional input units for bias. Adds 1 additional hidden unit for
     * bias
     */
    public RBFNeuralNet(int _I, int _J, int _K) {
        I = _I;
        J = _J;
        K = _K;

        z = new double[I];
        //z[I] = 1;//I corresponds to I + 1 in the literature
        u = new double[J][I];
        sigma = new double[J];
        y = new double[J + 1];
        y[J] = -1;//J corresponds to J + 1 in the literature
        w = new double[K][J + 1];
        o = new double[K];
    }

    @Override
    public double[] classify(double... inputPattern) throws UnequalArgsDimensionException {
        if (inputPattern.length != I)
            throw new UnequalArgsDimensionException();

        //assign input pattern to Z vector
        System.arraycopy(inputPattern, 0, z, 0, I);

        //activate each RBF (hidden unit)
        for (int j = 0; j < J; j++) {
            y[j] = new Gaussian().evaluate(distanceBetweenVectors(z, u[j]), sigma[j]);
        }

        //activate each output unit
        for (int k = 0; k < K; k++) {
            o[k] = sumProductVectors(w[k], y);
        }

        return Arrays.copyOf(o, o.length);
    }

    @Override
    public int getDimensionality() {
        //doubt I'll need this
        return 0;
    }

    @Override
    public double[] getWeightVector() {
        //rethink
        return null;
    }

    @Override
    public void setWeightVector(double... _weightVector) throws UnequalArgsDimensionException {
        //won't need this
    }

    /**
     * Euclidean distance between two vectors
     *
     * @return distance
     */
    public double distanceBetweenVectors(double[] v1, double[] v2) {
        double distance = 0;

        for (int i = 0; i < v1.length; i++) {
            distance += Math.pow(v2[i] - v1[i], 2);
        }

        return Math.sqrt(distance);
    }

    /**
     * Sum product of two vectors
     *
     * return sum product
     */
    private double sumProductVectors(double[] v1, double[] v2) {
        double sumProduct = 0;

        for (int i = 0; i < v1.length; i++) {
            sumProduct += v1[i] * v2[i];
        }

        return sumProduct;
    }

    public final int I;
    public final int J;
    public final int K;

    public final double[] z;
    public final double[][] u;
    public final double[] sigma;
    public final double[] y;
    public final double[][] w;
    public final double[] o;
}
