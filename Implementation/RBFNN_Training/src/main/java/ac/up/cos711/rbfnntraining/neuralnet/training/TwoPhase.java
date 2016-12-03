package ac.up.cos711.rbfnntraining.neuralnet.training;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.data.Pattern;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import ac.up.cos711.rbfnntraining.neuralnet.metric.ClassificationAccuracy;
import ac.up.cos711.rbfnntraining.neuralnet.metric.DefaultNetworkError;
import ac.up.cos711.rbfnntraining.neuralnet.metric.INetworkError;
import ac.up.cos711.rbfnntraining.neuralnet.util.ThresholdOutOfBoundsException;
import java.util.Iterator;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import ac.up.cos711.rbfnntraining.neuralnet.RBFNeuralNet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Implements the Two-Phase Learning algorithm for RBFNN
 *
 * @author Abrie van Aardt
 */
public class TwoPhase implements IFFNeuralNetTrainer {

    public TwoPhase() {
        ACCEPTABLE_TRAINING_ERROR = 0.1;//todo: find good value
        w_LEARNING_RATE = 0.01;//todo: find good value   
        init_LVQ_LEARNING_RATE = 0.01;
        neighbourhood_RADIUS = 1;
        MAX_EPOCH = 15;
        defaultNetworError = new DefaultNetworkError();
        classificationAccuracy = new ClassificationAccuracy();
        trainingErrorHistory = new double[MAX_EPOCH];
        validationErrorHistory = new double[MAX_EPOCH];
        epochs = new double[MAX_EPOCH];

        for (int i = 0; i < epochs.length; i++) {
            epochs[i] = i + 1;
        }
    }

    /**
     *
     * @param _acceptableTError
     * @param lvq_learningRate
     * @param w_learningRate
     * @param _classificationRigor
     * @param _maxEpoch
     * @throws ThresholdOutOfBoundsException
     */
    public TwoPhase(double _acceptableTError, double lvq_learningRate, double w_learningRate, int neighbourhood_Radius, double _classificationRigor, int _maxEpoch)
            throws ThresholdOutOfBoundsException {
        ACCEPTABLE_TRAINING_ERROR = _acceptableTError;
        w_LEARNING_RATE = w_learningRate;
        MAX_EPOCH = _maxEpoch;
        defaultNetworError = new DefaultNetworkError();
        classificationAccuracy = new ClassificationAccuracy(_classificationRigor);
        trainingErrorHistory = new double[MAX_EPOCH];
        validationErrorHistory = new double[MAX_EPOCH];
        trainingAccHistory = new double[MAX_EPOCH];
        validationAccHistory = new double[MAX_EPOCH];
        epochs = new double[MAX_EPOCH];
        init_LVQ_LEARNING_RATE = lvq_learningRate;
        lvq_LEARNING_RATE = init_LVQ_LEARNING_RATE;
        neighbourhood_RADIUS = neighbourhood_Radius;
    }

    @Override
    public void train(RBFNeuralNet network, Dataset trainingset, Dataset validationset)
            throws UnequalArgsDimensionException {

        Logger.getLogger(getClass().getName())
                .log(Level.INFO, "Started neural network training...");

        initialise(network, trainingset);

        double prevAvg;
        double avgValidationError = 0;
        double devValidationError = 0;
        double stdDevValidationError = 0;
        validationError = 0;
        double[] outputs = new double[1];
        double[] targets;

        int epoch = 0;
        long duration = System.nanoTime();

        neighbourhood_RADIUS = 0;//todo: adjust possibly

        do {
            //reset pattern clusters at the start of the epoch
            patternClusters = new ArrayList<>(network.J);

            //initialise patternClusters
            //can just index from here on 
            for (int j = 0; j < network.J; j++) {
                patternClusters.add(new ArrayList<>());
            }

            ++epoch;

            //prevent memorisation of pattern order
            trainingset.shuffle();
            Iterator<Pattern> patterns = trainingset.iterator();
            trainingError = 0;

            while (patterns.hasNext()) {
                Pattern p = patterns.next();
                targets = p.getTargets();
                outputs = network.classify(p.getInputs());
                trainingError += DefaultNetworkError.errorForPattern(targets, outputs);
                //todo: applied in succession within the same poch, might have to split up
                learningVectorQuantiser(network, p.getInputs());
                gradientDescent(network, targets, outputs);
            }

            updateSigmas(network);

            trainingError /= (trainingset.size() * outputs.length);

            //calculate running average of validation and standard deviation
            prevAvg = avgValidationError;
            validationError = defaultNetworError.measure(network, validationset);
            avgValidationError += (validationError - avgValidationError) / epoch;
            devValidationError += (validationError - avgValidationError) * (validationError - prevAvg);
            stdDevValidationError = Math.sqrt(devValidationError / epoch);

            trainingErrorHistory[epoch - 1] = trainingError;
            validationErrorHistory[epoch - 1] = validationError;
            trainingAccHistory[epoch - 1] = classificationAccuracy.measure(network, trainingset);
            validationAccHistory[epoch - 1] = classificationAccuracy.measure(network, validationset);

            Logger.getLogger(getClass().getName())
                    .log(Level.FINER, "Epoch {0}: E_t = {1}, E_v = {2}, E_v` = {3}, stdDev(E_v) = {4}",
                            new Object[]{
                                epoch,
                                String.format("%.4f", trainingError),
                                String.format("%.4f", validationError),
                                String.format("%.4f", avgValidationError),
                                String.format("%.4f", stdDevValidationError)
                            }
                    );

            //decay the LVQ-I learning rate
            //todo: Tau is 1, see if other values work better
            lvq_LEARNING_RATE = init_LVQ_LEARNING_RATE * Math.pow(Math.E, -(epoch / (0.1 * MAX_EPOCH)));
        }
        while (epoch < MAX_EPOCH);

        duration = System.nanoTime() - duration;

        Logger.getLogger(getClass().getName())
                .log(Level.INFO, "Training completed in {0} epoch(s) ({1}s) with "
                        + " E_t = {2}, E_v = {3}.",
                        new Object[]{
                            epoch,
                            duration / 1000000000,
                            String.format("%.4f", trainingError),
                            String.format("%.4f", validationError)
                        }
                );
    }

    /**
     * Initialize all weights
     *
     * @param network
     */
    private void initialise(RBFNeuralNet network, Dataset trainingset) {
        Iterator<Pattern> patterns = trainingset.iterator();
        int numPatterns = trainingset.size();
        List<Double> allInputsInComponentForm = new ArrayList<>(network.I * numPatterns);

        //calculate average of inputs in the training set        
        for (int n = 1; n <= numPatterns; n++) {
            Pattern pattern = patterns.next();
            double[] inputs = pattern.getInputs();

            for (int j = 0; j < network.J; j++) {
                for (int i = 0; i < network.I; i++) {
                    //set centre vector equal to running average
                    network.u[j][i] = calculateNewAverage(network.u[j][i], inputs[i], n);
                }
            }

            //extract input vectors component columns
            // to be used to initialise widths as stdDev
            for (int i = 0; i < network.I; i++) {
                allInputsInComponentForm.add(inputs[i]);
            }
        }

        //calculate standard deviation of the input
        //the second parameter to calculateSampleStdDev() is an average
        //the average here is the average in all dimensions of the input
        //any Uji can be used since they have been set to the average of the input
        double tempStdDev = calculateSampleStdDev(allInputsInComponentForm.stream().mapToDouble(d -> d).toArray(), Arrays.stream(network.u[0]).average().getAsDouble());

        //set all widths to standard deviation of inputs
        for (int j = 0; j < network.J; j++) {
            network.sigma[j] = tempStdDev;
        }

        for (int k = 0; k < network.K; k++) {
            for (int j = 0; j < network.J + 1; j++) {
                //initialise the hidden-to-output weights
                network.w[k][j] = rand.nextDouble() * 2.0 / Math.sqrt(network.J + 1.0) - 1.0 / Math.sqrt(network.J + 1);
            }
        }
    }

    /**
     * Adjusts only the hidden-to-output weights using gradient descent
     *
     * @param network
     * @param targets
     * @param outputs
     */
    private void gradientDescent(RBFNeuralNet network, double[] t, double[] o) {
        for (int k = 0; k < network.K; k++) {
            for (int j = 0; j < network.J + 1; j++) {
                //adjust hidden-to-output weights
                network.w[k][j] += w_LEARNING_RATE * ((t[k] - o[k]) * network.y[j]);
            }
        }
    }

    /**
     * Adjusts only the RBF centres and widths by performing clustering on the
     * input patterns
     *
     * @param network
     * @param trainingset
     * @param validationset
     */
    private void learningVectorQuantiser(RBFNeuralNet network, double[] z) {

        double minDistance = Double.MAX_VALUE;
        int winnerIndex = 0;

        for (int j = 0; j < network.J; j++) {
            double dist = network.distanceBetweenVectors(z, network.u[j]);

            if (dist < minDistance) {
                minDistance = dist;
                winnerIndex = j;
            }
        }

        //update centers of winner and neighbours
        for (int j = 0; j < network.J; j++) {
            if (Math.abs(j - winnerIndex) <= neighbourhood_RADIUS) {
                for (int i = 0; i < network.I; i++) {
                    network.u[j][i] += lvq_LEARNING_RATE * (z[i] - network.u[j][i]);
                }
            }
        }

        //add pattern to winning unit cluster
        patternClusters.get(winnerIndex).add(z);
    }

    private void updateSigmas(RBFNeuralNet network) {
        for (int j = 0; j < network.J; j++) {
            List<double[]> patterns = patternClusters.get(j);
            
            //set the average to zero
            network.sigma[j] = 0;

            //calculate the new average distance between centre j and patterns that belong to it
            for (int p = 1; p <= patterns.size(); p++) {
                network.sigma[j] = calculateNewAverage(network.sigma[j], network.distanceBetweenVectors(patterns.get(p - 1), network.u[j]), p);
            }

        }
    }

    /**
     * Calculates the running average assuming indexing starts at
     *
     * @param oldAvg
     * @param newValue
     * @param currentNumber
     * @return new average
     */
    protected double calculateNewAverage(double oldAvg, double newValue, int currentNumber) {
        if (currentNumber == 0)
            return 0;
        else
            return (oldAvg * (currentNumber - 1) + newValue) / currentNumber;
    }

    /**
     * Calculates the sample standard deviation of the array of values
     *
     * @param values
     * @param avg
     * @return sample standard deviation
     */
    protected double calculateSampleStdDev(double[] values, double avg) {
        double stdDev = Arrays.stream(values).map(i -> Math.pow(i - avg, 2)).sum() / (values.length - 1);
        return Math.sqrt(stdDev);
    }

    public double getValidationError() {
        return validationError;
    }

    public double getTrainingError() {
        return trainingError;
    }

    public double[] getTrainingErrorHistory() {
        return trainingErrorHistory;
    }

    public double[] getValidationErrorHistory() {
        return validationErrorHistory;
    }

    public double[] getTrainingAccHistory() {
        return trainingAccHistory;
    }

    public double[] getValidationAccHistory() {
        return validationAccHistory;
    }

    public double[] getEpochs() {
        return epochs;
    }

    private final Random rand = new Random(System.nanoTime());
    private final INetworkError defaultNetworError;
    private final INetworkError classificationAccuracy;//todo: could decide to use this as stopping condition
    private final double ACCEPTABLE_TRAINING_ERROR;
    private final double w_LEARNING_RATE;
    private final double init_LVQ_LEARNING_RATE;
    private double lvq_LEARNING_RATE;
    private int neighbourhood_RADIUS;
    private final int MAX_EPOCH;
    private double trainingError;
    private double validationError;
    private double[] trainingErrorHistory;
    private double[] validationErrorHistory;
    private double[] trainingAccHistory;
    private double[] validationAccHistory;
    private double[] epochs;

    //Two-Phase params
    private List<List<double[]>> patternClusters;
}
