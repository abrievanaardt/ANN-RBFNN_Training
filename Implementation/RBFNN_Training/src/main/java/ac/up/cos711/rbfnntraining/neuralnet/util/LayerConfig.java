package ac.up.cos711.rbfnntraining.neuralnet.util;

import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.function.Sigmoid;

/**
 * Encapsulates the information needed to build a single layer of the neural
 * network.
 * 
 * @author Abrie van Aardt
 */
public class LayerConfig {
    public Function activationFunction = new Sigmoid();
    public int weightCountPerNeuron = 0;
    public int neuronCount = 1;
}
