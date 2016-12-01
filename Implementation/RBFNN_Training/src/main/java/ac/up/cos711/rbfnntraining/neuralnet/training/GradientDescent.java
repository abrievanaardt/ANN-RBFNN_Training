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
import ac.up.cos711.rbfnntraining.neuralnet.RBFNeuralNetTest;

/**
 * Implements the Gradient Descent Learning algorithm for RBFNN
 *
 * @author Abrie van Aardt
 */
public class GradientDescent implements IFFNeuralNetTrainer {

    public GradientDescent() {
        ACCEPTABLE_TRAINING_ERROR = 0.1;//todo: find good value
        w_LEARNING_RATE = 0.01;//todo: find good value        
        u_LEARNING_RATE = 0.01;//todo: find good value 
        d_MAX = 0.5;//todo: probably have to initialise this taking number of RBFs and input range in account, try range 2/j
        sigma_LEARNING_RATE = 0.01;//todo: find good value        
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
     * @param w_learningRate
     * @param u_learningRate
     * @param sigma_learningRate
     * @param _classificationRigor
     * @throws ThresholdOutOfBoundsException
     */
    public GradientDescent(double _acceptableTError, double w_learningRate, double u_learningRate, double sigma_learningRate, double d_max, double _classificationRigor, int _maxEpoch)
            throws ThresholdOutOfBoundsException {
        ACCEPTABLE_TRAINING_ERROR = _acceptableTError;
        w_LEARNING_RATE = w_learningRate;
        u_LEARNING_RATE = u_learningRate;
        sigma_LEARNING_RATE = sigma_learningRate;
        d_MAX = d_max;
        MAX_EPOCH = _maxEpoch;
        defaultNetworError = new DefaultNetworkError();
        classificationAccuracy = new ClassificationAccuracy(_classificationRigor);
        trainingErrorHistory = new double[MAX_EPOCH];
        validationErrorHistory = new double[MAX_EPOCH];
        trainingAccHistory = new double[MAX_EPOCH];
        validationAccHistory = new double[MAX_EPOCH];
        epochs = new double[MAX_EPOCH];

       
    }

    @Override
    public void train(RBFNeuralNetTest network, Dataset trainingset, Dataset validationset)
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

        do {
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
                adjustWeights(network, targets, outputs);
            }

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
    
    public double[] getEpochs(){
        return epochs;
    }

    /**
     * Initialize all weights
     *
     * @param network
     */
    private void initialise(RBFNeuralNetTest network, Dataset trainingset) {
        trainingset.shuffle();
        Iterator<Pattern> patterns = trainingset.iterator();
        for (int j = 0; j < network.J; j++) {
            Pattern pattern = patterns.next();
            double[] inputs = pattern.getInputs();

            //set centre vector equal to the input vector
            System.arraycopy(inputs, 0, network.u[j], 0, inputs.length);

            //initialise the input bias
            //network.u[j][network.I] = rand.nextDouble() * 2.0 - 1.0;
            //calculate width for j
            network.sigma[j] = d_MAX / Math.sqrt(network.J);
        }

        for (int k = 0; k < network.K; k++) {
            for (int j = 0; j < network.J + 1; j++) {
                //initialise the hidden-to-output weights
                network.w[k][j] = rand.nextDouble() * 2.0 / Math.sqrt(network.J + 1.0) - 1.0 / Math.sqrt(network.J + 1);
            }
        }
    }

    /**
     * Adjust all the weights in the NN
     *
     * @param network
     * @param targets
     * @param outputs
     */
    private void adjustWeights(RBFNeuralNetTest network, double[] t, double[] o) {
        for (int k = 0; k < network.K; k++) {
            for (int j = 0; j < network.J + 1; j++) {
                //adjust hidden-to-output weights
                network.w[k][j] += w_LEARNING_RATE * ((t[k] - o[k]) * network.y[j]);
            }
        }
        for (int j = 0; j < network.J; j++) {
            //common calculation between centre and width updates, reuse
            double partialSum = 0;
            for (int k = 0; k < network.K; k++) {
                partialSum += (t[k] - o[k]) * network.w[k][j];
            }

            for (int i = 0; i < network.I; i++) {
                //adjust centre vectors
                network.u[j][i] += u_LEARNING_RATE * (1.0 / Math.pow(network.sigma[j], 2)) * (network.z[i] - network.u[j][i]) * network.y[j] * partialSum;
            }

            double distanceZtoU = 0;
            for (int i = 0; i < network.I; i++) {
                distanceZtoU += Math.pow(network.z[i] - network.u[j][i], 2);
            }

            //adjust width vector
            network.sigma[j] += sigma_LEARNING_RATE * (1.0 / Math.pow(network.sigma[j], 3)) * distanceZtoU * network.y[j] * partialSum;
        }
    }

    private final Random rand = new Random(System.nanoTime());
    private final INetworkError defaultNetworError;
    private final INetworkError classificationAccuracy;//todo: could decide to use this as stopping condition
    private final double ACCEPTABLE_TRAINING_ERROR;
    private final double w_LEARNING_RATE;
    private final double u_LEARNING_RATE;
    private final double sigma_LEARNING_RATE;
    private final double d_MAX;
    private final int MAX_EPOCH;
    private double trainingError;
    private double validationError;
    private double[] trainingErrorHistory;
    private double[] validationErrorHistory;
    private double[] trainingAccHistory;
    private double[] validationAccHistory;
    private double[] epochs;
}
