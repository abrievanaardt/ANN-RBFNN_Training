package ac.up.cos711.rbfnntraining.neuralnet.util;

/**
 *
 * @author Abrie van Aardt
 */
public class ZeroNeuronException extends Exception {
    @Override
    public String getMessage(){
        return "There has to be at least one neuron in a layer";
    }
}
