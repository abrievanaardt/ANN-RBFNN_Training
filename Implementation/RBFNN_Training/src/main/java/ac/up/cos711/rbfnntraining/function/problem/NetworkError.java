package ac.up.cos711.rbfnntraining.function.problem;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.function.problem.RealProblem;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import ac.up.cos711.rbfnntraining.neuralnet.metric.INetworkError;
import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;

/**
 * This class provides functionality to evaluate neural network classification
 * error given a weight vector. The network and accompanying dataset is
 * injected.
 *
 * @author Abrie van Aardt
 */
public class NetworkError extends RealProblem {

    public NetworkError(IRBFNeuralNet _network, Dataset _dataset, INetworkError _networkError, double xmin, double xmax) {        
        super(xmin, xmax, _network.getDimensionality());
        //todo: check if this actually works
        network = _network;//not deep copied
        dataset = _dataset;
        networkError = _networkError;        
    }

    /**
     * Evaluates the error of the network (loss function) given all the weights
     * that should be used in the network, together with the dataset.
     *
     * @param x the weight vector
     * @return network error
     * @throws UnequalArgsDimensionException
     */
    @Override
    public double evaluate(double... x) throws UnequalArgsDimensionException {
        if (x.length != network.getDimensionality())
            throw new UnequalArgsDimensionException();

        network.setWeightVector(x);
        return networkError.measure(network, dataset);
    }
    
    @Override
    public String getName(){
        String datasetName = dataset.getDatasetName().toLowerCase();
        datasetName = datasetName.substring(0, 1).toUpperCase() + datasetName.substring(1, datasetName.length());
        return super.getName() + " " + datasetName;
    }

    private final IRBFNeuralNet network;
    private final Dataset dataset;
    private final INetworkError networkError;

}
