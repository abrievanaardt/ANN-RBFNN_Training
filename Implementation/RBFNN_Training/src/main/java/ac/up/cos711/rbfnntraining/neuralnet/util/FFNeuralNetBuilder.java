package ac.up.cos711.rbfnntraining.neuralnet.util;

import ac.up.cos711.rbfnntraining.function.Function;
import ac.up.cos711.rbfnntraining.function.util.NotAFunctionException;
import ac.up.cos711.rbfnntraining.neuralnet.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class is used to specify a configuration for a feed-forward neural
 * network. A call to the {@link FFNeuralNetBuilder#build()} method will
 * instantiate a network with the desired configuration.
 * 
 * @author Abrie van Aardt
 */
public class FFNeuralNetBuilder {

    public FFNeuralNetBuilder() {

    }

    /**
     * Adds configuration information for an additional layer in the neural
     * network.
     *
     * @param neuronCount
     * @param activationFunction
     * @return FFNeuralNetBuilder
     * @throws NotAFunctionException
     */
    public FFNeuralNetBuilder addLayer(int neuronCount, Class activationFunction)
            throws NotAFunctionException, ZeroNeuronException {

        LayerConfig layerConfig = new LayerConfig();

        if (neuronCount < 1)
            throw new ZeroNeuronException();

        try {
            layerConfig.activationFunction = (Function) activationFunction.newInstance();
        }
        catch (InstantiationException | IllegalAccessException | ClassCastException e) {
            throw new NotAFunctionException();
        }

        if (config.layers.size() == 0) {//dealing with input layer
            layerConfig.weightCountPerNeuron = 0;
        }
        else {//dealing with hidden/output layer
            LayerConfig prevLayerConfig
                    = config.layers.get(config.layers.size() - 1);
            //add 1 additional weight to act as the bias for neurons in
            //this layer
            layerConfig.weightCountPerNeuron = prevLayerConfig.neuronCount + 1;
        }

        layerConfig.neuronCount = neuronCount;
        config.layers.add(layerConfig);

        Logger
                .getLogger(getClass().getName())
                .log(Level.INFO, "Configuring NN layer {0}: {1} neuron(s), {2} activation "
                        + "function.", new Object[]{
                    config.layers.size(),
                    layerConfig.neuronCount,
                    layerConfig.activationFunction.getClass().getSimpleName()
                });

        return this;
    }

    /**
     * Instantiate a feed forward neural network with the specified
     * configuration.
     *
     * @return IRBFNeuralNet
     */
    public IRBFNeuralNet build() {

        Logger
                .getLogger(getClass().getName())
                .log(Level.INFO, "Building feed-forward neural network"
                        + " according to specs.");

        //clear configuration to reuse this bulder instance
        FFNeuralNetConfig tempConfig = config;
        config = new FFNeuralNetConfig();

        return new RBFNeuralNet(tempConfig);
    }

    private FFNeuralNetConfig config = new FFNeuralNetConfig();
}
