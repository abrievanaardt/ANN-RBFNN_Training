package ac.up.cos711.rbfnntraining.neuralnet.metric;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.data.Pattern;
import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.function.SquaredError;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import ac.up.cos711.rbfnntraining.neuralnet.metric.INetworkError;
import java.util.Iterator;
import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;

/**
 * Network error that is used for Validation and Generalisation Tests. This
 * class implements the Mean{@link SquaredError} function, with respect to
 * {@link Sigmoid}, over the total patterns in the dataset.
 *
 * @author Abrie van Aardt
 */
public class DefaultNetworkError implements INetworkError {

    @Override
    public double measure(IRBFNeuralNet network, Dataset testingSet) throws UnequalArgsDimensionException {
        double error = 0;

        Iterator<Pattern> patterns = testingSet.iterator();
        double[] outputs = new double[1];
        double[] targets = new double[1];

        while (patterns.hasNext()) {

            Pattern p = patterns.next();
            outputs = network.classify(p.getInputs());
            targets = p.getTargets();
            error += errorForPattern(targets, outputs);
        }

        error /= (testingSet.size() * outputs.length);

        return error;
    }

    public static double errorForPattern(double[] targets, double[] outputs) throws UnequalArgsDimensionException {
        double sum = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            sum += outputError.evaluate(targets[i], outputs[i]);
        }
        return sum;
    }

    private static final Function outputError = new SquaredError();

}
