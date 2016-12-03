package ac.up.cos711.rbfnntraining.neuralnet.training;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;
import ac.up.cos711.rbfnntraining.neuralnet.RBFNeuralNet;

/**
 * Interface to the functionality to train a fully connected feed forward neural
 * network. Search/training algorithms intended to be used to search weight
 * space for minimal error must implement this interface.
 *
 * @author Abrie van Aardt
 */
public interface IFFNeuralNetTrainer {

    /**
     * This method will adjust the network weights to produce minimal error on
     * the training set passed as the second parameter. Care should be taken to
     * ensure only the training set of a larger data set is passed as a view by
     * calling
     * <pre>
     * trainingset.shuffle().split(ratio);
     * </pre> on a {@link Dataset} object in client code.
     *
     * @param network
     * @param trainingset
     * @param validationset
     * @throws UnequalArgsDimensionException
     */
    public void train(RBFNeuralNet network, Dataset trainingset, Dataset validationset)
            throws UnequalArgsDimensionException;
}
