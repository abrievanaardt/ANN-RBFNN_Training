package ac.up.cos711.rbfnntraining.neuralnet.metric;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;

/**
 *
 * @author Abrie van Aardt
 */
public interface INetworkError {
    public double measure(IRBFNeuralNet network, Dataset testingSet) throws UnequalArgsDimensionException;
}
