//package ac.up.cos711.rbfnntraining.neuralnet.training;
//
//import ac.up.cos711.rbfnntraining.data.Dataset;
//import ac.up.cos711.rbfnntraining.data.Pattern;
//import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
//import ac.up.cos711.rbfnntraining.neuralnet.Neuron;
//import ac.up.cos711.rbfnntraining.neuralnet.metric.ClassificationAccuracy;
//import ac.up.cos711.rbfnntraining.neuralnet.metric.DefaultNetworkError;
//import ac.up.cos711.rbfnntraining.neuralnet.metric.INetworkError;
//import ac.up.cos711.rbfnntraining.neuralnet.util.ThresholdOutOfBoundsException;
//import java.util.Iterator;
//import java.util.Random;
//import java.util.logging.Level;
//import java.util.logging.Logger;
//import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;
//
///**
// * Implements the BackPropagation algorithm, assuming the {@link Sigmoid} 
// * activation function for hidden and output nodes. It is important that the 
// * dataset be normalized for the active domain of Sigmoid.
// *
// * @author Abrie van Aardt
// */
//public class BackPropagation implements IFFNeuralNetTrainer {
//
//    public BackPropagation() {
//        ACCEPTABLE_TRAINING_ERROR = 0.1;//todo: find good value
//        LEARNING_RATE = 0.01;//todo: find good value
//        BIN_SIZE = 10;//todo: find good value
//        MAX_EPOCH = 15;
//        defaultNetworError = new DefaultNetworkError();
//        classificationAccuracy = new ClassificationAccuracy();
//        trainingErrorHistory = new double[MAX_EPOCH];
//        validationErrorHistory = new double[MAX_EPOCH];
//    }
//
//    /**
//     *
//     * @param _acceptableTError
//     * @param _learningRate
//     * @param _binSize
//     * @param _classificationRigor
//     * @throws ThresholdOutOfBoundsException
//     */
//    public BackPropagation(double _acceptableTError, double _learningRate, int _binSize, double _classificationRigor, int _maxEpoch)
//            throws ThresholdOutOfBoundsException {
//        ACCEPTABLE_TRAINING_ERROR = _acceptableTError;
//        LEARNING_RATE = _learningRate;
//        BIN_SIZE = _binSize;
//        MAX_EPOCH = _maxEpoch;
//        defaultNetworError = new DefaultNetworkError();
//        classificationAccuracy = new ClassificationAccuracy(_classificationRigor);
//        trainingErrorHistory = new double[MAX_EPOCH];
//        validationErrorHistory = new double[MAX_EPOCH];
//        trainingAccHistory = new double[MAX_EPOCH];
//        validationAccHistory = new double[MAX_EPOCH];
//    }
//
//    @Override
//    public void train(RBFNeuralNet network, Dataset trainingset, Dataset validationset)
//            throws UnequalArgsDimensionException {
//
//        Logger.getLogger(getClass().getName())
//                .log(Level.INFO, "Started neural network training...");
//
//        initialise(network);
//
//        double prevAvg;
//        double avgValidationError = 0;
//        double devValidationError = 0;
//        double stdDevValidationError = 0;
//        validationError = 0;
//        double[] outputs = new double[1];
//        double[] targets;
//
//        int epoch = 0;
//        int patternNumber = 1;
//        long duration = System.nanoTime();
//
//        do {
//            ++epoch;
//            
//            //prevent memorisation of pattern order
//            trainingset.shuffle();
//            Iterator<Pattern> patterns = trainingset.iterator();
//            trainingError = 0;            
//            patternNumber = 1;
//            
//            while (patterns.hasNext()) {
//                Pattern p = patterns.next();
//                targets = p.getTargets();
//                outputs = network.classify(p.getInputs());
//                trainingError += DefaultNetworkError.errorForPattern(targets, outputs);
//                backPropogateError(network, targets, outputs);
//                if (patternNumber % BIN_SIZE == 0)
//                    triggerWeightUpdates(network);
//                ++patternNumber;
//            }
//            
//            //if last few patterns did not fill a bin, trigger a weight update
//            if (trainingset.size() % BIN_SIZE != 0)
//                triggerWeightUpdates(network);
//            
//            trainingError /= (trainingset.size() * outputs.length);
//            
//            //calculate running average of validation and standard deviation
//            prevAvg = avgValidationError;
//            validationError = defaultNetworError.measure(network, validationset);
//            avgValidationError += (validationError - avgValidationError)/epoch;
//            devValidationError += (validationError - avgValidationError) * (validationError - prevAvg);            
//            stdDevValidationError = Math.sqrt(devValidationError/epoch);    
//            
//            trainingErrorHistory[epoch-1] = trainingError;
//            validationErrorHistory[epoch-1] = validationError;
//            trainingAccHistory[epoch-1] = classificationAccuracy.measure(network, trainingset);
//            validationAccHistory[epoch-1] = classificationAccuracy.measure(network, validationset);
//                        
//            Logger.getLogger(getClass().getName())
//                .log(Level.FINER, "Epoch {0}: E_t = {1}, E_v = {2}, E_v` = {3}, stdDev(E_v) = {4}",
//                        new Object[]{
//                            epoch,                            
//                            String.format("%.4f", trainingError),
//                            String.format("%.4f", validationError),
//                            String.format("%.4f", avgValidationError),
//                            String.format("%.4f", stdDevValidationError)
//                        }
//                );
//        }
//        while (epoch < MAX_EPOCH);
//
//        duration = System.nanoTime() - duration;      
//
//        Logger.getLogger(getClass().getName())
//                .log(Level.INFO, "Training completed in {0} epoch(s) ({1}s) with "
//                        + " E_t = {2}, E_v = {3}.",
//                        new Object[]{
//                            epoch,
//                            duration / 1000000000,
//                            String.format("%.4f", trainingError),
//                            String.format("%.4f", validationError)
//                        }
//                );
//    }
//    
//    public double getValidationError(){
//        return validationError;        
//    }
//    
//    public double getTrainingError(){
//        return trainingError;
//    }
//    
//    public double[] getTrainingErrorHistory(){
//        return trainingErrorHistory;
//    }
//    
//    public double[] getValidationErorrHistory(){
//        return validationErrorHistory;
//    }
//    
//    public double[] getTrainingAccHistory() {
//        return trainingAccHistory;
//    }
//
//    public double[] getValidationAccHistory() {
//        return validationAccHistory;
//    }
//    
//    private void backPropogateError(IRBFNeuralNet network, double[] targets, double[] outputs) {
//        //obtain neurons
//        Neuron[][] layers = network.getNetworkLayers();
//
//        //calculate error signals from output nodes
//        double[] errorSignals = new double[outputs.length];
//        for (int i = 0; i < outputs.length; i++) {
//            errorSignals[i] = -(targets[i] - outputs[i]) * (1 - outputs[i]) * outputs[i];
//        }
//
//        int biasIndex;
//        double[] newErrorSignals;
//
//        //iterate through layers, from last to second to update weights
//        //input layer is excluded since identity function is assumed
//        for (int i = layers.length - 1; i >= 1; i--) {
//            //used to capture error signals for the next layer
//            newErrorSignals = new double[layers[i - 1].length];
//
//            for (int j = 0; j < layers[i].length; j++) {
//                //adjust all weights excluding the bias
//                for (int k = 0; k < layers[i][j].getWeightCount() - 1; k++) {
//                    accumulateWeightDelta(layers, errorSignals, i, j, k, WeightType.NORMAL);
//                    updateErrorSignal(layers, newErrorSignals, errorSignals, i, j, k);
//                }
//                //now adjust the bias weight
//                biasIndex = layers[i][j].getWeightCount() - 1;
//                accumulateWeightDelta(layers, errorSignals, i, j, biasIndex, WeightType.BIAS);
//            }
//
//            //update error signals to be used for the next layer
//            errorSignals = newErrorSignals;
//        }
//
//    }
//
//    private void triggerWeightUpdates(IRBFNeuralNet network) {
//        //obtain neurons
//        Neuron[][] layers = network.getNetworkLayers();
//
//        //update each weight in the network with its corresponding delta
//        //that was accumulated over BIN_SIZE times
//        for (int i = 1; i < layers.length; i++) {
//            for (int j = 0; j < layers[i].length; j++) {
//                for (int k = 0; k < layers[i][j].getWeightCount(); k++) {
//                    layers[i][j].setWeight(k,
//                            layers[i][j].getWeightAt(k)
//                            + layers[i][j].getWeightDeltaAt(k));
//                    //reset weight delta for future use
//                    layers[i][j].setWeightDelta(k, 0);
//                }
//            }
//        }
//    }
//
//    private void accumulateWeightDelta(Neuron[][] layers, double[] errorSignals, int i, int j, int k, WeightType type) {
//        double oldWeightDelta;
//        double newWeightDelta;
//        oldWeightDelta = layers[i][j].getWeightDeltaAt(k);
//        newWeightDelta = -LEARNING_RATE * errorSignals[j];
//
//        if (type == WeightType.NORMAL) {
//            newWeightDelta *= layers[i - 1][k].getOutput();//input for weight_k
//        }
//        else if (type == WeightType.BIAS) {
//            newWeightDelta *= -1;//input = -1 for bias            
//        }
//
//        newWeightDelta += oldWeightDelta;
//
//        layers[i][j].setWeightDelta(k, newWeightDelta);
//    }
//
//    private void updateErrorSignal(Neuron[][] layers, double[] newErrorSignals, double[] errorSignals, int i, int j, int k) {
//        newErrorSignals[k] += layers[i][j].getWeightAt(k)
//                * errorSignals[j]
//                * (1 - layers[i - 1][k].getOutput())
//                * layers[i - 1][k].getOutput();
//    }
//
//    /**
//     * Initialize all weights to a value in the range
//     * <pre>
//     *  [-1/sqrt(fanin), 1/sqrt(fanin)]
//     * </pre> where fanin = # weights leading to the neuron.
//     *
//     * @param network
//     */
//    private void initialise(IRBFNeuralNet network) {
//        Neuron[][] layers = network.getNetworkLayers();
//        for (int i = 0; i < layers.length; i++) {
//            for (int j = 0; j < layers[i].length; j++) {
//                int fanin = layers[i][j].getWeightCount();
//                double range = 1.0 / Math.sqrt(fanin);
//                for (int k = 0; k < fanin; k++) {
//                    layers[i][j].setWeight(k, rand.nextDouble() * 2 * range - range);
//                }
//            }
//        }
//    }
//
//    private final Random rand = new Random(System.nanoTime());
//    private final INetworkError defaultNetworError;
//    private final INetworkError classificationAccuracy;//todo: could decide to use this as stopping condition
//    private final double ACCEPTABLE_TRAINING_ERROR;
//    private final double LEARNING_RATE;
//    private final int BIN_SIZE;
//    private final int MAX_EPOCH;
//    private double trainingError;
//    private double validationError;
//    private double[] trainingErrorHistory;
//    private double[] validationErrorHistory;
//    private double[] trainingAccHistory;
//    private double[] validationAccHistory;    
//    
//    private enum WeightType {
//        BIAS, NORMAL
//    };
//
//}
